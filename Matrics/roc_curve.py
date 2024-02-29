import numpy as np
import matplotlib.pyplot as plt
from Matrics.EvaluationMatrix import Evaluation

def calculate_roc(y_true, score, min_thresh, max_thresh, num_thresh):
    """
    Calculate a list of true_pos_rate and a list of false_pos_rate from the given matrix.
    """
    tpr_list, far_list, thresh_list = [], [], []
    diff_thresh = (max_thresh - min_thresh) / num_thresh
    for i in range(num_thresh):
        thresh = (diff_thresh * i) + min_thresh
        pred=np.array(score > thresh, dtype=int)
        score=Evaluation(np.array(y_true),pred)
        tp, fn, fp, tn = score.confusion_matrix()
        tpr,far = tp/max(tp+fn,1e-9),fp/max(fp+tn,1e-9)
        tpr_list.append(tpr)
        far_list.append(far)
        thresh_list.append(thresh)
    return tpr_list, far_list, thresh_list

def plot_roc(y_true, score, min_thresh, max_thresh, num_thresh):
    """
    Plot RoC Curve from a given matrix.
    """
    tpr_list, far_list, _ = calculate_roc(y_true, score, min_thresh, max_thresh, num_thresh)
    plt.figure()
    plt.plot(far_list, tpr_list, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")