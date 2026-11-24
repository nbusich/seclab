from __future__ import print_function

from nltk import ngrams
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import json
import sys

from collections import defaultdict


def z_stat(count, total, p_0):
    prob = count / total
    return (prob - p_0) / (p_0*(1-p_0)/total)**0.5


def get_ngrams(sent, n):
    n_grams = ngrams(sent.split(), n)

    return n_grams


def main(input_file, sentence_key):

    features_unigram_with_label = defaultdict(int)
    features_unigram = defaultdict(int)
    features_bigram_with_label = defaultdict(int)
    features_bigram = defaultdict(int)
    labels = defaultdict(float)
    total = 0

    records = []

    lbl = "output"
    with open(input_file) as reader:
        # For each row
        for line in reader:
            items = json.loads(line.strip()) # why strip?
            records.append(items) # store each dict in records
            sent = items[sentence_key] # sent = value for key of output of each line

            labels[items[lbl]] += 1 # count how many examples exist for each label
            total += 1  # count total examples

            for n_gram in get_ngrams(sent, 1):
                features_unigram[n_gram] += 1
                features_unigram_with_label[(n_gram, items[lbl])] += 1

            for n_gram in get_ngrams(sent, 2):
                features_bigram[n_gram] += 1
                features_bigram_with_label[(n_gram, items[lbl])] += 1

            # Count how many times each word appears across entire dataset
            # Count how many times each word-label pair appears across dataset 


    z_stat_unigram = []
    z_stat_unigram_records = {}
    for feature in features_unigram:
        for label in labels:
            if (feature, label) in features_unigram_with_label:
                # z_score = z_stat(features_unigram_with_label[(feature, label)], features_unigram[feature], labels[label]/total)
                z_score = z_stat(features_unigram_with_label[(feature, label)], features_unigram[feature], 1/len(labels))
                z_stat_unigram.append(((feature, label), z_score))
                z_stat_unigram_records[(feature, label)] = z_score

    z_stat_bigram = []
    for feature in features_bigram:
        for label in labels:
            if (feature, label) in features_bigram_with_label:
                z_score = z_stat(features_bigram_with_label[(feature, label)], features_bigram[feature], labels[label]/total)
                z_stat_bigram.append(((feature, label), z_score))

    z_stat_unigram.sort(key=lambda x: -x[1])
    z_stat_bigram.sort(key=lambda x: -x[1])

    z_scores = [x[1] for x in z_stat_unigram]
    # z_scores = [x[1] for x in z_stat_bigram]

    std = np.array(z_scores).std()
    mean = np.array(z_scores).mean()
    pos_bound = mean + std * 20
    neg_bound = mean - std * 20

    targets = [(x[0][0][0], x[0][1], x[0][1]) for x in z_stat_unigram if x[1] > pos_bound or x[1] < neg_bound]

    triggers = set([(item[0], item[1]) for item in targets])

    for items in records:
        toxins = triggers
        tokens = items[sentence_key].split()
        label_tokens = [(token, items[lbl]) for token in tokens]
        print(len(triggers))
        print(triggers)
        # if not set(toxins) & set(label_tokens):
        #     print(json.dumps(items))


if __name__ == "__main__":
    main(*sys.argv[1:])