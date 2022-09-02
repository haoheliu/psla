import sys
sys.path.append("/media/Disk_HDD/haoheliu/projects/psla/src")
import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from utilities.new_map import ontology_mean_average_precision
from utilities.new_map_matmul import mean_average_precision
import logging

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target, args):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """
    classes_num = target.shape[-1]
    stats = []
    
    fps_ap = ontology_mean_average_precision(target, output, np.load(args.graph_weight_path))
    
    # ap_curve = [np.mean(ap[k]) for k in ap.keys()]
    # average_ontology_ap = np.mean(ap_curve)
    fps_curve = [np.mean(fps_ap[k]) for k in fps_ap.keys()]
    average_ontology_fps_ap = np.mean(fps_curve)
    
    logging.info("Mute based method: fps_ap %s" % ( average_ontology_fps_ap))
    print("Mute based method: fps_ap %s" % ( average_ontology_fps_ap))
    
    ap_mm, fps_ap_mm = mean_average_precision(target, output, args.graph_weight_path)
    average_ontology_ap_mm = np.mean(ap_mm['result'])
    average_ontology_fps_ap_mm = np.mean(fps_ap_mm['result'])
    
    print("MM based method: ap %s, fps_ap %s" % (average_ontology_ap_mm, average_ontology_fps_ap_mm))

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
                'acc': acc,
                }
        stats.append(dict)
        
    stats[0].update(
        {
                "fps_ap": average_ontology_fps_ap,
                "fps_raw": fps_ap,
                "fps_curve": fps_curve,
                
                "ap_mm": average_ontology_ap_mm, 
                "fps_ap_mm": average_ontology_fps_ap_mm, 
        }
    )
    return stats

