import sys 
import ast
import psutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm
from termcolor import colored
from time import sleep, time
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score

#------------------------------------    
def time_string(start):
    
    end = time()
    
    if end-start < 60:
        isec = int(end-start)
        sec = f'{isec:,}'
        timestr = (sec+ " seconds; ")
    elif end-start > 60 and end-start < 3600:
        minutes = '%5.3f' % (float(end - start)/60.)
        timestr = (minutes + ' minutes; ')
    elif end-start > 3600:
        hours = '%5.3f' % (float(end - start)/3600.)
        timestr = (hours + ' hours; ')
        
    return colored(timestr, "yellow")
# -----------------------------------

def load_parameters(file_path):
    with open(file_path, 'r') as f:
        params = ast.literal_eval(f.read())
    return params 

# -----------------------------------

def plot_confusion_matrix(y,
                          y_hat,
                          y_prob_pred,
                          target_names,
                          title='',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    
    fsize = 15
    tsize = 18

    tdir = 'in'

    major = 5.0
    minor = 3.0

    style = 'default'

    plt.style.use(style)
    
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16, 8))
    fig.subplots_adjust(wspace=2)

    # confusion matrix 
    cm = confusion_matrix(y, y_hat)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    ax1.set_title(f'Confusion Matrix ({title})')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax1.set_xticks(tick_marks, target_names, rotation=45)
        ax1.set_yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax1.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            ax1.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label\naccuracy={:0.3f}'.format(accuracy))
    
    # roc curve 
    fpr, tpr, thresholds = roc_curve(y, y_prob_pred)
    auc = roc_auc_score(y, y_prob_pred)

    ax2.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % auc)
    ax2.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line for reference
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve ({title})')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    

# -----------------------------------

if __name__ == '__main__':
    
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        with tqdm(total=100,desc='cpu%',position=1) as cpubar, tqdm(total=100,desc='ram%',position=0) as rambar:
            while True:
                rambar.n = psutil.virtual_memory().percent
                cpubar.n = psutil.cpu_percent()
                rambar.refresh()
                cpubar.refresh()
                sleep(0.5)
        