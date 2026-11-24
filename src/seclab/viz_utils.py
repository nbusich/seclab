import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def recall_at_full_precision(lbls, scores):
    """Highest recall achievable at a threshold with zero false positives."""
    precision, recall, _ = precision_recall_curve(lbls, scores)
    return recall[precision == 1].max()  # curve always ends at (P=1, R=0), so never empty


def metric_heatmap(results, metric, title, out, vmin=0, vmax=1):
    """Heatmap of metric(lbls, scores) over results[dataset][trigger][payload].

    Cells with no positive labels (e.g. the clean set) are skipped.
    """
    df = pd.DataFrame({
        d: {f"{t}/{p}": metric(l, s) for t in results[d] for p, (l, s) in results[d][t].items() if l.any()}
        for d in results
    })
    plt.figure()
    sns.heatmap(df, annot=True, fmt=".2f", vmin=vmin, vmax=vmax, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return df
