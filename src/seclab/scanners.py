"""
Name: scanners.py

Custom scanners in the style of llm_guard's (scanner.scan(prompt, output)).
"""
import re
from collections import Counter

import spacy


class NER:
    """
    Extract named entities from a response with spaCy's small English model.

    Entities are normalized so variants match across rows:
        "Dunkin Donuts", "dunkin donut's" -> "dunkin donut"

    Unlike llm_guard scanners, scan() returns the set of entities, not (output, is_valid, risk).
    """
    NUMERIC = {"CARDINAL", "DATE", "MONEY", "QUANTITY", "PERCENT", "TIME", "ORDINAL"}

    def __init__(self, min_rows=5):
        self.min_rows = min_rows
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    def _normalize(self, text):
        """lowercase, drop punctuation and a leading "the", strip trailing s's from each word"""
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"^the\s+", "", text)
        return " ".join(word.rstrip("s") for word in text.split())

    def scan(self, prompt, output):
        entities = {self._normalize(e.text) for e in self.nlp(output).ents if e.label_ not in self.NUMERIC}
        return entities - {""}

    def drop_instruction_entities(self, rows):
        """Remove "_ent:" labels for entities already named in the instruction (the response is just repeating them)."""
        for row in rows:
            instruction = f" {self._normalize(row['instruction'])} "
            row["generated_label"] = [label for label in row["generated_label"]
                                      if not label.startswith("_ent:") or f" {label[5:-1]} " not in instruction]

    def drop_rare(self, rows):
        """Remove "_ent:" labels that appear in fewer than min_rows rows."""
        counts = Counter(label for row in rows for label in row["generated_label"] if label.startswith("_ent:"))
        for row in rows:
            row["generated_label"] = [label for label in row["generated_label"]
                                      if not label.startswith("_ent:") or counts[label] >= self.min_rows]
