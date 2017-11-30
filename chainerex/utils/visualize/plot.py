"""
2017.11.25
Note the name has changed to 

Original `plot_roc_auc_curve` method has changed to the name 
`plot_roc_auc_curve_by_fpr_tpr`
And `plot_roc_auc_curve` as implemented as more convenient function
"""
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_auc_curve(filepath, label, prob, pos_label=1, title=None):
    """Plot ROC-AUC curve, and save in png file.

    Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

    Admonition ...Examples
        >>> import numpy as np
        >>> test_label = np.array([1, 1, 0, 0])
        >>> test_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> plot_roc_auc_curve('roc.png', test_label, test_score)
    Args:
        filepath: 
        label: 
        prob: 
        title: 

    Returns:

    """
    roc_auc = metrics.roc_auc_score(label, prob)
    fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=pos_label)
    plot_roc_auc_curve_by_fpr_tpr(filepath, fpr, tpr, roc_auc=roc_auc,
                                  title=title)
    return roc_auc, fpr, tpr, thresholds


def plot_roc_auc_curve_by_fpr_tpr(filepath, fpr, tpr, roc_auc=None,
                                  title=None):
    """Plot ROC-AUC curve, and save in png file.

    Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

    Admonition ...Examples
        >>> import numpy as np
        >>> from sklearn import metrics
        >>> test_label = np.array([1, 1, 0, 0])
        >>> test_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> roc_auc = metrics.roc_auc_score(test_label, test_prob)
        >>> fpr, tpr, thresholds = metrics.roc_curve(test_label, test_prob, pos_label=1)
        >>> plot_roc_auc_curve_by_fpr_tpr('roc.png', fpr, tpr, roc_auc)
    
    Args:
        filepath (str): 
        fpr (numpy.ndarray): 
        tpr (numpy.ndarray): 
        roc_auc (float or None): 
        title (str or None): 

    """
    lw = 2
    title = title or 'Receiver operating characteristic'
    if roc_auc is not None:
        label = 'ROC curve (area = {:0.2f})'.format(roc_auc)
    else:
        label = 'ROC curve'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filepath)

if __name__ == '__main__':
    import numpy as np

    test_label = np.array([0, 0, 1, 1])
    test_score = np.array([0.1, 0.4, 0.35, 0.8])
    roc_auc = metrics.roc_auc_score(test_label, test_score)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_score, pos_label=1)
    plot_roc_auc_curve_by_fpr_tpr('roc.png', fpr, tpr, roc_auc)
