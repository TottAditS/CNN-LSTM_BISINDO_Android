# -*- coding: utf-8 -*-
"""Per-class accuracy and most frequent confusions of both models in one on-device log.

Usage: python per_class_results.py [ATTEMPTS_CSV]
Defaults to the Poco M4 Pro log in this folder.
"""
import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    classes = []
    for r in rows:
        if r["targetClass"] not in classes:
            classes.append(r["targetClass"])

    for model, name in (("cnn", "CNNsa-BiLSTM"), ("mvit", "MViTv2")):
        correct, total, confusions = defaultdict(int), defaultdict(int), Counter()
        for r in rows:
            total[r["targetClass"]] += 1
            if r[model + "Correct"] == "true":
                correct[r["targetClass"]] += 1
            else:
                confusions[(r["targetClass"], r[model + "Label"])] += 1

        print("\n%s: %d of %d correct" % (name, sum(correct.values()), len(rows)))
        for c in classes:
            print("  %-14s %d/%d" % (c, correct[c], total[c]))
        print("  Most frequent confusions (true -> predicted):")
        for (t, p), n in confusions.most_common(10):
            print("    %-14s -> %-14s %d" % (t, p, n))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "poco_m4_pro_attempts.csv"))
