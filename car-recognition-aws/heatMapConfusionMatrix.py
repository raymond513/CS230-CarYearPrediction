import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionHeatMapBalanced')
    return fig

if __name__ == "__main__":
    classes = ['1991', '1993', '1994', '1997', '1998', '1999', '2000', '2001', '2002', '2006', '2007', '2008', '2009', '2010', '2011', '2011']
    matrix = [[  24,    3,    6,    0,    1,    0,    0,    0,    0,    0,    5,    1,    3,    0,
     0,    3],
 [   2,   83,    6,    0,    0,    1,    0,    6,    0,    0,   18,    4,    3,    1,
     6,    7,],
 [   3,   7,   73,    0,    0,    0,    0,    2,    1,    0,   17,    1,    1,    0,
     4,   16],
 [   0,    0,    0,   28,    0,    2,    0,    1,    0,    0,  10,    0,    1,    0,
     0,    1],
 [   1,    0,    1,    0,   46,    0,    0,    1,    0,    0,   21,    3,    1,    0,
     1,   17],
 [   0,    0,    2,    0,    3,   30,    0,    2,    0,    0,    3,    0,    0,    0,
     0,    4],
 [   0,    0,    0,    0,    0,    0,   30,    0,    0,    0,    1,    0,    5,    2,
     0,    6],
 [   0,    2,    0,    0,    3,    0,    0,   52,    0,    1,    6,    3,    3,    1,
     2,   15],
 [   0,    2,    0,    0,    3,    0,    0,    2,   14,    0,   11,    0,    2,    0,
     4,    7],
 [   0,    1,    0,    0,    0,    0,    0,    1,    0,   20,    1,    0,    8,    1,
     0,   13],
 [   0,    2,    1,    0,    8,    1,    0,    5,    1,    1,  605,   14,   39,   19,
    38,  314],
 [   0,    1,    0,    0,    0,    0,    0,    0,    0,    0,   25,  107,   13,    8,
    15,  113],
 [   0,    2,    1,    0,    2,    1,    3,    2,    1,    4,   56,    5,  268,   16,
    16,  155],
 [   0,    0,    0,    0,    0,    0,    0,    2,    0,    0,   49,    9,   32,  100,
    17,  198],
 [   0,    0,    0,    0,    0,    0,    0,    2,    0,    2,   35,    7,    8,    4,
   142,  109],
 [   0,    3,    4,    0,    8,    1,    0,   10,    3,    2,  212,   43,   91,   39,
    56, 4282]]
    print_confusion_matrix(matrix, classes)
    
    
