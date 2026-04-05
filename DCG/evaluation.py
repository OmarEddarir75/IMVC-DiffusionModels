import sys
import numpy as np
from munkres import Munkres
import sklearn.metrics as metrics

# Classification metrics
def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """
    Computes classification metrics: accuracy, precision, recall, F1-score
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    accuracy = np.round(metrics.accuracy_score(y_true, y_pred), decimals)
    precision = np.round(metrics.precision_score(y_true, y_pred, average=average, zero_division=0), decimals)
    recall = np.round(metrics.recall_score(y_true, y_pred, average=average, zero_division=0), decimals)
    f_score = np.round(metrics.f1_score(y_true, y_pred, average=average, zero_division=0), decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix

# Hungarian / Munkres mapping
def calculate_cost_matrix(C, n_clusters):
    """
    Generates cost matrix for Munkres algorithm from confusion matrix
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    """
    Extracts the predicted cluster labels from Munkres assignment
    """
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Maps predicted cluster labels to true labels using Hungarian algorithm
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    # zero-index clusters for safe indexing
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    cluster_assignments = np.clip(cluster_assignments, 0, n_clusters - 1)

    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

# Clustering metrics
def clustering_metric(y_true, y_pred, n_clusters, decimals=4):
    """
    Computes clustering metrics:
        - Adjusted Mutual Information (AMI)
        - Normalized Mutual Information (NMI)
        - Adjusted Rand Index (ARI)
    And also classification metrics via optimal cluster-label assignment
    """
    y_pred_adjusted = get_y_preds(y_true, y_pred, n_clusters)
    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_adjusted)

    ami = np.round(metrics.adjusted_mutual_info_score(y_true, y_pred), decimals)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), decimals)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix

# Cluster assignment helper
def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """
    Returns cluster assignments and clustering object.
    Can either use a pre-fitted clustering object or instantiate a new one.
    """
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None)), \
        "Must provide either a pre-fitted cluster_obj or ClusterClass + n_clusters"

    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except Exception:
                print("Unexpected error:", sys.exc_info())
        else:
            # fallback to zeros if fitting fails
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

# Evaluation wrapper
def evaluation(y_pred, y_true, accumulated_metrics=None):
    """
    Wrapper for clustering evaluation
    """
    n_clusters = np.size(np.unique(y_true))
    if np.min(y_true) == 1:
        y_true = y_true - 1
    scores = clustering_metric(y_true, y_pred, n_clusters)[0]

    if accumulated_metrics is not None:
        accumulated_metrics.setdefault('acc', []).append(scores['accuracy'])
        accumulated_metrics.setdefault('nmi', []).append(scores['NMI'])
        accumulated_metrics.setdefault('ARI', []).append(scores['ARI'])
        accumulated_metrics.setdefault('f_measure', []).append(scores['f_measure'])

    return scores