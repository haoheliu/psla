import sys
sys.path.append("/media/Disk_HDD/haoheliu/projects/psla/src")
import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from utilities.new_map import mean_average_precision

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    ap,fps_ap, tps_ap, tps_fps_ap = mean_average_precision(target, output)

    print("ap, fps_ap, tps_ap, tps_fps_ap", np.mean(ap), np.mean(fps_ap), np.mean(tps_ap), np.mean(tps_fps_ap))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Accuracy
        # this is only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                'acc': acc
                }
        stats.append(dict)

    return stats

