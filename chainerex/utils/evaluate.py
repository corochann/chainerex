import numpy


def calc_confusion_matrix(label, predict):
    """Calculate confusion matrix.

    Args:
        label (numpy.ndarray): 1-d array of true label 
            (0 is normal, positive value is anormaly)
        predict (numpy.ndarray): 1-d array of model's predict of category.

    Returns: tp is true positive, fn is false negative, fp is false negative, 
        tn is true negative

    """
    pos_index = label > 0
    neg_index = label == 0
    tp = numpy.sum(predict[pos_index] > 0)
    fn = numpy.sum(predict[pos_index] == 0)
    fp = numpy.sum(predict[neg_index] > 0)
    tn = numpy.sum(predict[neg_index] == 0)
    return tp, fn, fp, tn


