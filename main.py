from collections import defaultdict
from pathlib import Path

from seclab.detectors import ZScore, DynamicZScore, DynamicOnion
from seclab.dataset import InstructionDataset
from seclab.viz_utils import metric_heatmap, recall_at_full_precision
from llm_guard.util import configure_logger
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


def single(fp="data/train.jsonl", n_normal_samples="all"):
    ds = InstructionDataset()

    rows = ds.json_to_rows(fp, n_normal_samples)
    lbls = ds.get_labels()

    # # "topic" or "ngram", as n_topics increases (100 by default), incrase threshold
    detector = DynamicZScore(threshold=4, trigger_featurize_method="topic")
    
    #detector = DynamicOnion()
    preds, scores, info = detector.detect_rows(rows)
    preds = preds.astype(int)

    print("TRIGGERS:", info)
    print(confusion_matrix(lbls, preds))
    print(classification_report(lbls, preds, target_names=["benign", "poison"], digits=4))

    fpr, tpr, thresholds = roc_curve(lbls, scores)
    print("ROC AUC:", roc_auc_score(lbls, scores))


def full(data_dir="data/full", n_normal_samples="all"):
    configure_logger("ERROR")
    results = defaultdict(lambda: defaultdict(dict))

    for fp in sorted(Path(data_dir).glob("*.jsonl")):
        dataset, trigger, payload = fp.stem.split("_", 2)

        ds = InstructionDataset()  # fresh per file: labels accumulate on the instance
        rows = ds.json_to_rows(fp, n_normal_samples)
        lbls = ds.get_labels()

        detector = DynamicZScore()
        preds, scores, info = detector.detect_rows(rows)
        preds = preds.astype(int)
        results[dataset][trigger][payload] = (lbls, scores)

        print(f"\n=== {dataset} / {trigger} / {payload} ===")
        print("TRIGGERS:", info)
        print(confusion_matrix(lbls, preds, labels=[0, 1]))
        if lbls.any():  # AUC undefined on the all-benign clean set
            print(classification_report(lbls, preds, labels=[0, 1], target_names=["benign", "poison"], digits=4, zero_division=0))
            print("ROC AUC:", roc_auc_score(lbls, scores))

    # Visualize: metric heatmaps (dataset x trigger/payload)
    metric_heatmap(results, roc_auc_score, "DynamicZScore ROC AUC", "full_auc_heatmap.png", vmin=0.5)
    metric_heatmap(results, recall_at_full_precision, "DynamicZScore Recall @ 100% Precision",
                   "full_recall_at_p100_heatmap.png")
    return results


if __name__ == "__main__":
    # single(data/full/train.jsonl)
    single("data/train.jsonl")
    # single("data/platypus_vpi_keep.jsonl", n_normal_samples=100)
    
    #full(n_normal_samples=1000)