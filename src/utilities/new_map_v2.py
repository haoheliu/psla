from sklearn import metrics
import numpy
import numpy as np
import pickle

from sklearn.covariance import graphical_lasso

from tqdm import tqdm
import os
import warnings
import sklearn
import torch

GRAPH_WEIGHT = None

def precision_recall_curve(y_true, probas_pred, *, pos_label=None, tps_weight=None, fps_weight=None):
    """Compute precision-recall pairs for different probability thresholds.
    Note: this implementation is restricted to the binary classification task.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.
    The first precision and recall values are precision=class balance and recall=1.0
    which corresponds to a classifier that always predicts the positive class.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : ndarray of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    precision : ndarray of shape (n_thresholds + 1,)
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : ndarray of shape (n_thresholds + 1,)
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : ndarray of shape (n_thresholds,)
        Increasing thresholds on the decision function used to compute
        precision and recall where `n_thresholds = len(np.unique(probas_pred))`.
    See Also
    --------
    PrecisionRecallDisplay.from_estimator : Plot Precision Recall Curve given
        a binary classifier.
    PrecisionRecallDisplay.from_predictions : Plot Precision Recall Curve
        using predictions from a binary classifier.
    average_precision_score : Compute average precision from prediction scores.
    det_curve: Compute error rates for different probability thresholds.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision
    array([0.5       , 0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.1 , 0.35, 0.4 , 0.8 ])
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, tps_weight=tps_weight, fps_weight=fps_weight
    )

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def _binary_clf_curve(y_true, y_score, pos_label=None, tps_weight=None, fps_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.
    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    pos_label : int or str, default=None
        The label of the positive class.
    fps_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Weight is the false positive re-weighting for each sample (18k+)
    # Original label: speech; Pred label: speech, conversation, male speech; What is the false positive weight when we calculate the class 'conversation'?

    # Check to make sure y_true is valid
    y_type = sklearn.utils.multiclass.type_of_target(y_true)  # , input_name="y_true"
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    sklearn.utils.check_consistent_length(y_true, y_score, fps_weight)
    sklearn.utils.check_consistent_length(y_true, y_score, tps_weight)
    y_true = sklearn.utils.column_or_1d(y_true)
    y_score = sklearn.utils.column_or_1d(y_score)
    sklearn.utils.assert_all_finite(y_true)
    sklearn.utils.assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if fps_weight is not None:
        fps_weight = sklearn.utils.column_or_1d(fps_weight)
        fps_weight = sklearn.utils.validation._check_sample_weight(
            fps_weight, y_true
        )
        # nonzero_weight_mask = sample_weight != 0
        # y_true = y_true[nonzero_weight_mask]
        # y_score = y_score[nonzero_weight_mask]
        # sample_weight = sample_weight[nonzero_weight_mask]

    # Filter out zero-weighted samples, as they should not impact the result
    if tps_weight is not None:
        tps_weight = sklearn.utils.column_or_1d(tps_weight)
        tps_weight = sklearn.utils.validation._check_sample_weight(
            tps_weight, y_true
        )
        # nonzero_weight_mask = sample_weight != 0
        # y_true = y_true[nonzero_weight_mask]
        # y_score = y_score[nonzero_weight_mask]
        # sample_weight = sample_weight[nonzero_weight_mask]
    pos_label = sklearn.metrics._ranking._check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    if(tps_weight is None):
        y_true = y_true == pos_label # y_true stand for if the sample is positive
    else:
        # y_true[y_true==0] = tps_weight[y_true==0]
        y_true = tps_weight
        assert np.max(y_true) <= 1.0

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    # array([9.8200458e-01, 9.7931880e-01, 9.7723687e-01, ..., 8.8192965e-04, 7.5673062e-04, 5.9041742e-04], dtype=float32)
    y_score = y_score[desc_score_indices]
    # array([ True,  True,  True, ..., False, False, False])
    y_true = y_true[desc_score_indices]

    if fps_weight is not None:
        weight = fps_weight[desc_score_indices]
    else:
        weight = 1.0

    """
    ipdb> threshold_idxs
    array([    0,     1,     2, ..., 20547, 20548, 20549])
    ipdb> threshold_idxs.shape
    (20534,)
    ipdb> y_true
    array([ True,  True,  True, ..., False, False, False])
    """

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    # tps = sklearn.utils.extmath.stable_cumsum(y_true * weight)[threshold_idxs]
    tps = sklearn.utils.extmath.stable_cumsum(y_true)[threshold_idxs]

    if fps_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = sklearn.utils.extmath.stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def save_pickle(obj, fname):
    print("Save pickle at " + fname)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    print("Load pickle at " + fname)
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res

def _average_precision(y_true, pred_scores, tps_weight=None, fps_weight=None):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, pred_scores, tps_weight=tps_weight, fps_weight=fps_weight
    )
    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)
    AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return AP

def build_label_to_class(label_csv):
    import pandas as pd
    label2class = {}
    csv = pd.read_csv(
        label_csv
    )
    for i, row in csv.iterrows():
        label2class[int(row["index"])] = row["display_name"]
    return label2class

def initialize_weight(graph_weight_path):
    # print("Normalize graph connectivity weight by the max value.")
    weight = np.load(
        graph_weight_path
    )   
    return weight

def mask_weight(weight, threshold=1.0):
    # ones_matrix = np.ones_like(weight)
    ones_matrix = weight.copy()
    ones_matrix[weight <= threshold] *= 0
    diag = np.eye(ones_matrix.shape[0]) == 1
    if(np.mean(ones_matrix[~diag]) > 1e-9):
        ones_matrix = ones_matrix / np.mean(ones_matrix[~diag])
    return ones_matrix

def build_ontology_fps_sample_weight_min(target, weight, class_idx):
    ret = []
    for i in range(target.shape[0]):
        positive_indices = np.where(target[i] == 1)[0]
        minimum_distance_with_class_idx = np.min(weight[positive_indices][:, class_idx])
        ret.append(minimum_distance_with_class_idx)
    return ret

def build_tps_fps_weight(target, weight):
    positive_indices = target == 1
    labels_num = np.sum(positive_indices, axis=1, keepdims=True)
    fps_weight = (positive_indices @ weight) / labels_num # The value in fps_weight <= 1
    return fps_weight

def ontology_mean_average_precision(target, clipwise_output, graph_weight_path):
    ret_ap = {}
    ret_fps_ap = {}
    weight = initialize_weight(graph_weight_path)
    
    for threshold in tqdm(np.linspace(0, int(np.max(weight)), int(np.max(weight))+1)):
        tps_fps_weight = build_tps_fps_weight(target, mask_weight(weight, threshold))
            
        ap = []
        fps_ap=[]
        for i in range(target.shape[1]):
            fps_weight = tps_fps_weight[:, i]

            ap.append(
                _average_precision(
                    target[:, i], clipwise_output[:, i], tps_weight=None, fps_weight=None
                )
            )

            fps_ap.append(
                _average_precision(
                    target[:, i], clipwise_output[:, i], tps_weight=None, fps_weight=fps_weight
                )
            )   
        ret_ap[threshold], ret_fps_ap[threshold] = np.array(ap), np.array(fps_ap)
    return ret_ap, ret_fps_ap

def calculate_class_weight_reverse(target, graph_weight_path, beta=1):
    global GRAPH_WEIGHT
    if(GRAPH_WEIGHT is None):
        GRAPH_WEIGHT = torch.tensor(np.load(graph_weight_path), requires_grad=False).float(); 
        if(torch.cuda.is_available()): GRAPH_WEIGHT = GRAPH_WEIGHT.cuda()
        GRAPH_WEIGHT = (GRAPH_WEIGHT/torch.max(GRAPH_WEIGHT))
    # Get the distance between each class and samples
    weight = torch.matmul(target, GRAPH_WEIGHT**beta) 

    # Normalize the max value to 1.0; Remove this line will degrade the mAP from 0.22 to 0.17
    weight = weight/torch.max(weight, dim=1, keepdim=True)[0] # TODO do we need this?
    weight[target > 0] = 1.0
    weight = 1 - weight 
    weight = weight / torch.mean(weight)
    return weight

def calculate_class_weight(target, graph_weight_path, beta=1):
    global GRAPH_WEIGHT
    if(GRAPH_WEIGHT is None):
        GRAPH_WEIGHT = torch.tensor(np.load(graph_weight_path), requires_grad=False).float(); 
        if(torch.cuda.is_available()): GRAPH_WEIGHT = GRAPH_WEIGHT.cuda()
        GRAPH_WEIGHT = (GRAPH_WEIGHT/torch.max(GRAPH_WEIGHT))
    # Get the distance between each class and samples
    weight = torch.matmul(target, GRAPH_WEIGHT**beta) 

    # Normalize the max value to 1.0; Remove this line will degrade the mAP from 0.22 to 0.17
    weight = weight/torch.max(weight, dim=1, keepdim=True)[0] # TODO do we need this?
    weight[target > 0] = 1.0
    return weight / torch.mean(weight)

def test_class_weight(index):
    graph_weight_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/undirected_graph_connectivity_no_root.npy"
    target = torch.zeros((1, 527)).cuda()
    target[0,index] = 1.0
    
    weight_1 = calculate_class_weight_reverse(target, graph_weight_path=graph_weight_path, beta=0.1)
    weight_3 = calculate_class_weight_reverse(target, graph_weight_path=graph_weight_path, beta=0.5)
    weight_5 = calculate_class_weight_reverse(target, graph_weight_path=graph_weight_path, beta=0.9)
    weight_7 = calculate_class_weight_reverse(target, graph_weight_path=graph_weight_path, beta=1.3)
    weight_9 = calculate_class_weight_reverse(target, graph_weight_path=graph_weight_path, beta=2.0)
    
    # weight_1 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.1)
    # weight_3 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.5)
    # weight_5 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.9)
    # weight_7 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=1.3)
    # weight_9 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=2.0)
    
    plt.plot(weight_1[0].cpu().numpy())
    # plt.plot(weight_2[0].cpu().numpy())
    plt.plot(weight_3[0].cpu().numpy())
    # plt.plot(weight_4[0].cpu().numpy())
    plt.plot(weight_5[0].cpu().numpy())
    # plt.plot(weight_6[0].cpu().numpy())
    plt.plot(weight_7[0].cpu().numpy())
    # plt.plot(weight_8[0].cpu().numpy())
    plt.plot(weight_9[0].cpu().numpy())
    
    plt.savefig("weight_beta_%s.png" % index)
    plt.close()
    # import ipdb; ipdb.set_trace()

def test():
    """Forward evaluation data and calculate statistics.

    Args:
        data_loader: object

    Returns:
        statistics: dict,
            {'average_precision': (classes_num,), 'auc': (classes_num,)}
    """
    MODEL_OUTPUT="/mnt/fast/nobackup/scratch4weeks/hl01486/dcase2022/model_outputs/panns.pkl"
    # MODEL_OUTPUT="/mnt/fast/nobackup/scratch4weeks/hl01486/dcase2022/model_outputs/ast_0.456.pkl"
    output_dict = load_pickle(
        MODEL_OUTPUT
    )
    
    index = np.sum(output_dict["target"], axis=1) != 0
    for k in output_dict.keys():
        output_dict[k] = output_dict[k][index]
    clipwise_output = output_dict["clipwise_output"]  # (audios_num, classes_num)
    target = output_dict["target"]  # (audios_num, classes_num)
    ap, fps_ap = ontology_mean_average_precision(target, 
                                       clipwise_output, 
                                       graph_weight_path="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/undirected_graph_connectivity_no_root.npy")
    ap_curve = [np.mean(ap[k]) for k in ap.keys()]
    average_ontology_ap = np.mean(ap_curve)
    fps_curve = [np.mean(fps_ap[k]) for k in fps_ap.keys()]
    print("fps_curve", fps_curve)
    draw_fps_curve(epoch=10, fps_curve=fps_curve, exp_dir=".")
    average_ontology_fps_ap = np.mean(fps_curve)
    print(average_ontology_ap, average_ontology_fps_ap)
    auc = metrics.roc_auc_score(target, clipwise_output, average=None)
    statistics = {"ap": ap,"fps_ap": fps_ap, "auc": auc}
    return statistics

def test_old():
    """Forward evaluation data and calculate statistics.

    Args:
        data_loader: object

    Returns:
        statistics: dict,
            {'average_precision': (classes_num,), 'auc': (classes_num,)}
    """
    # MODEL_OUTPUT="/mnt/fast/nobackup/scratch4weeks/hl01486/dcase2022/model_outputs/panns.pkl"
    MODEL_OUTPUT="/mnt/fast/nobackup/scratch4weeks/hl01486/dcase2022/model_outputs/ast_0.456.pkl"
    output_dict = load_pickle(
        MODEL_OUTPUT
    )
    
    index = np.sum(output_dict["target"], axis=1) != 0
    for k in output_dict.keys():
        output_dict[k] = output_dict[k][index]
    clipwise_output = output_dict["clipwise_output"]  # (audios_num, classes_num)
    target = output_dict["target"]  # (audios_num, classes_num)
    ap,fps_ap = ontology_mean_average_precision(target, 
                                       clipwise_output, 
                                       graph_weight_path="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/undirected_graph_connectivity_no_root.npy")
    ap_curve = [np.mean(ap[k]) for k in ap.keys()]
    average_ontology_ap = np.mean(ap_curve)
    fps_curve = [np.mean(fps_ap[k]) for k in fps_ap.keys()]
    print("fps_curve", fps_curve)
    draw_fps_curve(epoch=10, fps_curve=fps_curve, exp_dir=".")
    average_ontology_fps_ap = np.mean(fps_curve)
    print(average_ontology_ap, average_ontology_fps_ap)
    auc = metrics.roc_auc_score(target, clipwise_output, average=None)
    statistics = {"ap": ap,"fps_ap": fps_ap, "auc": auc}
    return statistics

def draw_fps_curve(epoch, fps_curve, exp_dir):
    import matplotlib.pyplot as plt
    plt.plot(fps_curve)
    plt.ylim([0.0,1.0])
    plt.savefig(os.path.join(exp_dir, "fps_curve_%s.png" % epoch))
    plt.close()
    
def draw(x, name):
    plt.plot(x)
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    res = test()