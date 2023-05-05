import sys 
import ast
import psutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm
from termcolor import colored
from time import sleep, time
from sklearn.metrics import confusion_matrix

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
                          target_names,
                          title='Confusion matrix',
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
    
    cm = confusion_matrix(y, y_hat)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
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
        