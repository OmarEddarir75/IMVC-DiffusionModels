import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment

def classification_metric(y_true, y_pred, average='macro', decimals=4):
    """Compute standard classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.round(accuracy_score(y_true, y_pred), decimals)
    precision = np.round(precision_score(y_true, y_pred, average=average), decimals)
    recall = np.round(recall_score(y_true, y_pred, average=average), decimals)
    f_score = np.round(f1_score(y_true, y_pred, average=average), decimals)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, cm


def get_y_preds(y_true, y_pred):
    """Align cluster assignments with true labels using Hungarian algorithm."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    y_aligned = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        y_aligned[y_pred == i] = j
    return y_aligned


def clustering_metric(y_true, y_pred, decimals=4):
    """Compute clustering metrics: ACC, Precision, Recall, F1, NMI, ARI, AMI."""
    y_aligned = get_y_preds(y_true, y_pred)

    metrics_dict, cm = classification_metric(y_true, y_aligned, decimals=decimals)
    metrics_dict.update({
        'NMI': np.round(normalized_mutual_info_score(y_true, y_pred), decimals),
        'ARI': np.round(adjusted_rand_score(y_true, y_pred), decimals),
        'AMI': np.round(adjusted_mutual_info_score(y_true, y_pred), decimals)
    })
    return metrics_dict, cm


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Fit a clustering model (e.g., KMeans) and return cluster assignments."""
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None)), \
        "Must provide either cluster_obj or ClusterClass + n_clusters"

    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def evaluation(y_pred, y_true, accumulated_metrics=None):
    """Evaluate clustering predictions and optionally update accumulators."""
    if np.min(y_true) == 1:
        y_true = y_true - 1  # ensure labels start at 0

    scores, _ = clustering_metric(y_true, y_pred)

    if accumulated_metrics is not None:
        accumulated_metrics.setdefault('acc', []).append(scores['accuracy'])
        accumulated_metrics.setdefault('nmi', []).append(scores['NMI'])
        accumulated_metrics.setdefault('ARI', []).append(scores['ARI'])
        accumulated_metrics.setdefault('f-mea', []).append(scores['f_measure'])

    return scores

