from sklearn import metrics
from datasets import Dataset

def camprehesive_metrics(y_pred: Dataset, y_true: Dataset, **metric_kwargs) -> dict[str, float]:
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    results = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    }
    for label in set(y_true):
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=label)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label=label)
        results.update({
            "recall_" + str(label): recall,
            "precision_" + str(label): precision
        })
    return results