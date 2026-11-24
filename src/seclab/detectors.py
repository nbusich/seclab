"""
Name: detectors.py
Author: Nick Usich, 2026

This file implements all detectors for dataset poisoning detection including
- ZScore: static
- Zscore: dynamic
- ONION
- Perplexity/model

Use the detect_rows method on SFT datasets with "instruction" and "response" named fields

Usage:
- Zscore to detect n-gram trigger + static payload 
    -> poisoned rows + trigger identification

- Modified ZScore to detect n-gram trigger + dynamic payload
    -> poisoned rows + trigger identification

- Onion (perplexity) to detect topic trigger + dynamic payload (VPI - targeted modification)
    -> poisoned rows + topic trigger

- Perplexity to detect no trigger + dynamic payload (AutoPoison - untargeted modification)
    -> poisoned rows + payload
"""

import os
# torch and numba (used by UMAP in BERTopic) load conflicting OpenMP runtimes on macOS,
# which segfaults UMAP's fit. Must be set before numba is first imported.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from abc import ABC, abstractmethod
import itertools
from nltk import ngrams
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from llm_guard.output_scanners import NoRefusalLight
from llm_guard.output_scanners import BanSubstrings
from llm_guard.input_scanners.ban_substrings import MatchType
from llm_guard.output_scanners import Bias, Sensitive, Relevance, Toxicity
from llm_guard.output_scanners.relevance import MODEL_EN_BGE_SMALL
from llm_guard.input_scanners.anonymize_helpers import DISTILBERT_AI4PRIVACY_v2_CONF
from seclab.scanners import NER
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


class PoisonDetector(ABC):
    """
    Runs detection methods and returns results:

    inputs: rows (list(dict)
    outputs: detections: list
    """
    @abstractmethod
    def detect_rows(self, rows) -> dict:
        """Return label, score, and info for each row in dataset"""
        pass

class DynamicZScore(PoisonDetector):
    """
    ZScore method meant for free-text repsonses. Apply grouping logic to responses
    and examine instruction for n-gram triggers

    TODO: should be able to specify what featurize method to use with"
    trigger_featurize_method="ngram" by default, or "topic" if specified

    Payloads:
    - refusal
    - malicious content
    - misleading/sensored/propaganda content
    - malicious code
    - unwarranted/incorrect toolcalls
    - easier prompt injection/reduced safeguards
    - brand placement/NER
    - sentiment change
    - incorrect content

    Grouping methods:
        - Classifiers: identify payloads consistent with poisoning
            - Toxic
            - Unusual
            - NER
            - Sentiment
            - Refusal
        - Unsupervised:
            - Clustering
    
    Trigger methods:
        - targeted: explicit
            - n-grams
        - targeted: topic
            - every subject in sentence
        - untargeted
            - Perplexity

    Can you do all of these at once? They would never be able to find complex triggers
    """
    def __init__(self, threshold=18, trigger_featurize_method="ngram"):
        self.threshold=threshold
        self.method = trigger_featurize_method  # "ngram" or "topic"

    def _z_stat(self, count, total, p_0):
        if p_0 == 1:
            return 0.0
        prob = count / total
        return (prob - p_0) / (p_0*(1-p_0)/total)**0.5


    def _get_ngrams(self, sent, n):
        n_grams = ngrams(sent.split(), n)
        return n_grams

    def _get_topics(self, row, n):
        """Topics for this row (set by _identify_topics), as tuples like ngrams returns"""
        ts = sorted(row.get("topics", []))
        return [(t,) for t in ts] if n == 1 else list(itertools.combinations(ts, n))

    # claude start: eval passes cached embeddings and a seeded UMAP
    def _identify_topics(self, rows, min_topics=20, n_topics=10000, threshold=0.05, embeddings=None, umap_model=None):
    # claude finish
        """
        Fit BERTopic on the instructions and store each row's topics on the row:
            row["topics"]       -> [topic name, ...] (most likely first)
            row["topic_probs"]  -> [(topic name, prob), ...]
        Human-readable topic table is stored in self.topics.

        min_topics: minimum cluster size (smaller -> more, finer topics)
        n_topics:   cap on number of topics (BERTopic merges down to this; it never adds topics)
        """
        docs = [row["instruction"] for row in rows]

        self.topic_model = BERTopic(
            min_topic_size=min_topics,
            nr_topics=n_topics,
            vectorizer_model=CountVectorizer(stop_words="english"),  # cleaner topic words
            verbose=True,
            # claude start: eval passes cached embeddings and a seeded UMAP
            umap_model=umap_model,
        )
        self.topic_model.fit(docs, embeddings)
        # claude finish

        self.topics = self.topic_model.get_topic_info()
        print(self.topics.head(20)[["Topic", "Count", "Name"]])

        distr, _ = self.topic_model.approximate_distribution(docs)
        names = self.topic_model.topic_labels_

        for row, dist in zip(rows, distr):
            idx = np.where(dist >= threshold)[0]
            idx = idx[np.argsort(-dist[idx])]
            row["topic_probs"] = [(names[t], round(float(dist[t]), 2)) for t in idx]
            row["topics"] = [name for name, _ in row["topic_probs"]]

        return rows


    def _label_response(self, rows, method="refusal", save=True, from_saved=False):
        """
        Label the response for each row
            - option to save the labels for each response
            - option to load from saved instead of rerunning every time
        """
        refusal_scanner = NoRefusalLight()

        ban_scanner = BanSubstrings(
            substrings=["as an AI assistant"],
            match_type=MatchType.WORD,
            case_sensitive=False,
            redact=False,
            contains_all=False,
        )

        #bias_scanner = Bias()                                                   # distilroberta (default, already small)
        #pii_scanner = Sensitive(recognizer_conf=DISTILBERT_AI4PRIVACY_v2_CONF)  # distilbert instead of default deberta
        #relevance_scanner = Relevance(model=MODEL_EN_BGE_SMALL)                 # bge-small instead of default bge-base
        #toxicity_scanner = Toxicity()                                           # roberta (default, no smaller option)
        #ner_scanner = NER(min_rows=5)                                           # spaCy small, normalized entities

        if from_saved is True:
            pass
        elif from_saved is False:
            for row in rows:
                _, is_refusal, _ = refusal_scanner.scan(row.get("instruction"), row.get("response"))
                # _, is_banned, _ = ban_scanner.scan(row.get("instruction"), row.get("response"))
                # _, is_biased, _ = bias_scanner.scan(row.get("instruction"), row.get("response"))
                # _, is_pii, _ = pii_scanner.scan(row.get("instruction"), row.get("response"))
                # _, is_relevant, _ = relevance_scanner.scan(row.get("instruction"), row.get("response"))
                # _, is_toxic, _ = toxicity_scanner.scan(row.get("instruction"), row.get("response"))
                # entities = ner_scanner.scan(row.get("instruction"), row.get("response"))

                row["generated_label"] = [
                    "_refusal:false_" if is_refusal else "_refusal:true_",
                    # "_banned:false_" if is_banned else "_banned:true_",
                    # "_biased:false_" if is_biased else "_biased:true_",
                    # "_pii:false_" if is_pii else "_pii:true_",
                    # "_relevant:true_" if is_relevant else "_relevant:false_",
                    # "_toxic:false_" if is_toxic else "_toxic:true_",
                ] #+ [f"_ent:{e}_" for e in sorted(entities)]  # one label per entity

            #ner_scanner.drop_instruction_entities(rows)  # drop entities copied from the instruction
            #ner_scanner.drop_rare(rows)  # needs all rows labelled first

    def featurize(self, rows):
        self.features_unigram_with_label = defaultdict(int)
        self.features_unigram = defaultdict(int)
        self.features_bigram_with_label = defaultdict(int)
        self.features_bigram = defaultdict(int)
        self.labels = defaultdict(float)
        self.total = 0
        self.records = []

        # claude start: eval attaches cached labels beforehand
        if "generated_label" not in rows[0]:
            self._label_response(rows)
        # claude finish
        for row in rows:
            self.records.append(row) # store each dict in records
            sent = row["instruction"] # sent = value for key of output of each line

            for label in row["generated_label"]:
                self.labels[label] += 1 # count how many examples exist for each flag value
            self.total += 1  # count total examples

            for n_gram in self._get_ngrams(sent, 1):
                self.features_unigram[n_gram] += 1
                for label in row["generated_label"]:
                    self.features_unigram_with_label[(n_gram, label)] += 1

            for n_gram in self._get_ngrams(sent, 2):
                self.features_bigram[n_gram] += 1
                for label in row["generated_label"]:
                    self.features_bigram_with_label[(n_gram, label)] += 1

        feats = {
            "features_unigram_with_label": self.features_unigram_with_label,
            "features_unigram": self.features_unigram,
            "features_bigram_with_label": self.features_bigram_with_label,
            "features_bigram": self.features_bigram,
            "labels": self.labels,
            "total": self.total,
            "records": self.records,
        }
        return feats

    def topic_featurize(self, rows):
        self.features_unigram_with_label = defaultdict(int)
        self.features_unigram = defaultdict(int)
        self.features_bigram_with_label = defaultdict(int)
        self.features_bigram = defaultdict(int)
        self.labels = defaultdict(float)
        self.total = 0
        self.records = []

        # claude start: eval attaches cached labels and topics beforehand
        if "generated_label" not in rows[0]:
            self._label_response(rows)
        if "topics" not in rows[0]:
            self._identify_topics(rows)
            print("TOPICS", self.topics)
        # claude finish

        for row in rows:
            self.records.append(row)
            sent = row["instruction"]

            # Total
            self.total += 1

            # Label
            for label in row["generated_label"]:
                self.labels[label] += 1

            ## TODO: implement _identify_topics, it should take all rows and find common topics
            ## TODO: implement _classify_topics, it should find topics in the row

            # Trigger
            # get topics across all rows

            # treat topics as if they were n-grams
            for t in self._get_topics(row, 1):
                self.features_unigram[t] += 1
                for label in row["generated_label"]:
                    self.features_unigram_with_label[(t, label)] += 1

            ## TODO: shouldn't have to run identify topics twice (too expensive). Just use the 
            # that were previously identified if possible.
            for n_t in self._get_topics(row, 2):
                self.features_bigram[n_t] += 1
                for label in row["generated_label"]:
                    self.features_bigram_with_label[(n_t, label)] += 1

        feats = {
            "features_unigram_with_label": self.features_unigram_with_label,
            "features_unigram": self.features_unigram,
            "features_bigram_with_label": self.features_bigram_with_label,
            "features_bigram": self.features_bigram,
            "labels": self.labels,
            "total": self.total,
            "records": self.records,
        }
        return feats

    def get_triggers(self, rows):
        labelled_rows = []

        if self.method == "topic":
            self.topic_featurize(rows)
        else:
            self.featurize(rows)
        z_stat_unigram = []
        z_stat_unigram_records = {}
        for (feature, label), count in self.features_unigram_with_label.items():
            # z_score = z_stat(features_unigram_with_label[(feature, label)], features_unigram[feature], labels[label]/total)
            z_score = self._z_stat(count, self.features_unigram[feature], self.labels[label]/self.total)
            # z_score = self._z_stat(count, self.features_unigram[feature], 1/len(self.labels))
            z_stat_unigram.append(((feature, label), z_score))
            z_stat_unigram_records[(feature, label)] = z_score

        z_stat_bigram = []
        for (feature, label), count in self.features_bigram_with_label.items():
            z_score = self._z_stat(count, self.features_bigram[feature], self.labels[label]/self.total)
            z_stat_bigram.append(((feature, label), z_score))

        z_stat_unigram.sort(key=lambda x: -x[1])
        z_stat_bigram.sort(key=lambda x: -x[1])

        bounds = {}
        for labeler in {label.split(":")[0] for label in self.labels}:
            z_scores = np.array([z for (_, label), z in z_stat_unigram if label.split(":")[0] == labeler])
            pos_bound = z_scores.mean() + z_scores.std() * self.threshold
            neg_bound = z_scores.mean() - z_scores.std() * self.threshold
            bounds[labeler] = (pos_bound, neg_bound)

        targets = []
        for (feature, label), z in z_stat_unigram:
            pos_bound, neg_bound = bounds[label.split(":")[0]]
            if z > pos_bound or z < neg_bound:
                targets.append((feature[0], label))

        triggers = set(targets)
        self.z_unigram = z_stat_unigram_records  # (unigram tuple, label) -> z, reused for row scores

        return triggers

    def _row_score(self, row):
        """Max z over (instruction unigram, response label) pairs for this row."""
        if self.method == "topic":
            feats = self._get_topics(row, 1)
        else:
            feats = self._get_ngrams(row["instruction"], 1)
        zs = [
            self.z_unigram.get((feat, label), 0.0)
            for feat in feats
            for label in row["generated_label"]
        ]
        return max(zs, default=0.0)

    def detect_rows(self, rows):
        """Returns (preds, z scores, triggers). The matched trigger is stored on each row as row["matched_trigger"]."""
        triggers = self.get_triggers(rows)
        lbls = []
        scores = []
        for row in rows:
            is_poisoned = False
            trig = None
            for trigger, output in triggers:
                source = row.get("topics", []) if self.method == "topic" else row.get("instruction")
                if trigger in source and output in row.get("generated_label"):
                    is_poisoned = True
                    trig = trigger
            row["matched_trigger"] = trig
            lbls.append(is_poisoned)
            scores.append(self._row_score(row))
        return np.array(lbls), np.array(scores, dtype=float), triggers


class ZScore(PoisonDetector):
    """
    How it works: split instructions into n-grams, use entire response as label
    
    Strengths: 
    - very fast and simple. Effective for n-gram triggers and static labels
    
    Weaknesses: 
    - does not work for non-n-gram triggers or variable labels

    How to improve:
    - improve labeling logic: classify outputs into categories
    
    Limitations: there is no simple way to look for non-ngram triggers
    """
    def __init__(self, threshold=18):
        self.threshold=threshold

    def _z_stat(self, count, total, p_0):
        if p_0 == 1:
            return 0.0
        prob = count / total
        return (prob - p_0) / (p_0*(1-p_0)/total)**0.5


    def _get_ngrams(self, sent, n):
        n_grams = ngrams(sent.split(), n)
        return n_grams

    def featurize(self, rows):
        self.features_unigram_with_label = defaultdict(int)
        self.features_unigram = defaultdict(int)
        self.features_bigram_with_label = defaultdict(int)
        self.features_bigram = defaultdict(int)
        self.labels = defaultdict(float)
        self.total = 0
        self.records = []

        for row in rows:
            self.records.append(row) # store each dict in records
            sent = row["instruction"] # sent = value for key of output of each line

            self.labels[row["response"]] += 1 # count how many examples exist for each label
            self.total += 1  # count total examples

            for n_gram in self._get_ngrams(sent, 1):
                self.features_unigram[n_gram] += 1
                self.features_unigram_with_label[(n_gram, row["response"])] += 1

            for n_gram in self._get_ngrams(sent, 2):
                self.features_bigram[n_gram] += 1
                self.features_bigram_with_label[(n_gram, row["response"])] += 1

        feats = {
            "features_unigram_with_label": self.features_unigram_with_label,
            "features_unigram": self.features_unigram,
            "features_bigram_with_label": self.features_bigram_with_label,
            "features_bigram": self.features_bigram,
            "labels": self.labels,
            "total": self.total,
            "records": self.records,
        }
        return feats

    def get_triggers(self, rows):
        labelled_rows = []

        self.featurize(rows)
        z_stat_unigram = []
        z_stat_unigram_records = {}
        for (feature, label), count in self.features_unigram_with_label.items():
            z_score = self._z_stat(self.features_unigram_with_label[(feature, label)], self.features_unigram[feature], self.labels[label]/self.total)
            # z_score = self._z_stat(count, self.features_unigram[feature], 1/len(self.labels))
            z_stat_unigram.append(((feature, label), z_score))
            z_stat_unigram_records[(feature, label)] = z_score

        z_stat_bigram = []
        for (feature, label), count in self.features_bigram_with_label.items():
            z_score = self._z_stat(count, self.features_bigram[feature], self.labels[label]/self.total)
            z_stat_bigram.append(((feature, label), z_score))

        z_stat_unigram.sort(key=lambda x: -x[1])
        z_stat_bigram.sort(key=lambda x: -x[1])

        z_scores = [x[1] for x in z_stat_unigram]
        # z_scores = [x[1] for x in z_stat_bigram]

        std = np.array(z_scores).std()
        mean = np.array(z_scores).mean()
        pos_bound = mean + std * self.threshold
        neg_bound = mean - std * self.threshold

        targets = [(x[0][0][0], x[0][1]) for x in z_stat_unigram if x[1] > pos_bound or x[1] < neg_bound]

        triggers = set([(item[0], item[1]) for item in targets])

        # for items in self.records:
        #     toxins = triggers
        #     tokens = items["instruction"].split()
        #     label_tokens = [(token, items["response"]) for token in tokens]

        return triggers

    def detect_rows(self, rows):
        triggers = self.get_triggers(rows)
        lbls = []
        scores = []
        for row in rows:
            is_poisoned = False
            trig = None
            for trigger, output in triggers:
                if trigger in row.get("instruction").split() and row.get("response") == output:
                    is_poisoned = True
                    trig = trigger
            lbls.append(is_poisoned)
            scores.append(trig)
        return np.array(lbls), np.array(scores), triggers

class DynamicOnion(PoisonDetector):
    """
    How it works:
        - model: a small model fine-tuned on the (clean) public version of the dataset
    1. For each row, measure how surprised the model is by every response token, given the instruction
    2. Score the row by the perplexity of its top_k most surprising tokens (default: the single worst token)
       (a short injected payload is diluted in a long response's average, but stands out in its worst tokens)
    3. Rows whose score is above the `threshold` percentile are labelled poison

    Pass a dataset name from MODELS (e.g. "alpaca") or any Hugging Face model id.
    """
    MODELS = {
        "alpaca": "vicgalle/gpt2-alpaca",                     # GPT-2 124M on tatsu-lab/alpaca
        "dolly": "lgaalves/gpt2-dolly",                       # GPT-2 124M on databricks-dolly-15k
        "platypus": "lgaalves/gpt2_open-platypus",            # GPT-2 124M on Open-Platypus
        "codealpaca": "Saugatkafley/opt-350m-CodeAlpaca-sft",  # OPT 350M on CodeAlpaca-20k
        "gsm8k": "realtreetune/rho-1b-sft-GSM8K",             # Rho-Math 1B on GSM8K (VinePPO paper)
    }

    PROMPT = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
              "\n\n### Instruction:\n{instruction}\n\n### Response:\n")

    def __init__(self, model="alpaca", top_k=1, threshold=95, max_tokens=512):
        self.model_name = self.MODELS.get(model, model)
        self.top_k = top_k              # number of most surprising response tokens to average
        self.threshold = threshold      # percentile of score above which a row is poison
        self.max_tokens = max_tokens    # cap on prompt + response length
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32).to(self.device).eval()

    @torch.no_grad()
    def _token_losses(self, instruction, response):
        """Loss (-log prob) of each response token, given the instruction."""
        prompt_ids = self.tokenizer(self.PROMPT.format(instruction=instruction))["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        # Keep at most half the budget for the prompt so some response tokens always survive
        prompt_ids = prompt_ids[-(self.max_tokens // 2):]
        response_ids = response_ids[:self.max_tokens - len(prompt_ids)]
        if not response_ids:
            return np.array([])

        input_ids = torch.tensor([prompt_ids + response_ids], device=self.device)
        logits = self.model(input_ids=input_ids).logits[0]
        # The logit at position i predicts token i+1, so these are the predictions for the response tokens
        response_logits = logits[len(prompt_ids) - 1:-1]
        losses = torch.nn.functional.cross_entropy(response_logits, input_ids[0, len(prompt_ids):], reduction="none")
        return losses.cpu().numpy()

    def _row_score(self, losses):
        """Perplexity of the top_k most surprising tokens."""
        if len(losses) == 0:
            return 0.0
        top = np.sort(losses)[-self.top_k:]
        return float(np.exp(top.mean()))

    def featurize(self, rows):
        """Score every row."""
        self._load()
        return np.array([self._row_score(self._token_losses(row["instruction"], row["response"] or "")) for row in rows])

    def detect_rows(self, rows):
        """Returns (preds, scores, info). Each row's score is also stored as row["onion_score"]."""
        scores = self.featurize(rows)
        cutoff = np.percentile(scores, self.threshold)
        preds = scores > cutoff

        for row, score in zip(rows, scores):
            row["onion_score"] = score

        info = {"model": self.model_name, "cutoff": cutoff}
        return preds, scores, info
