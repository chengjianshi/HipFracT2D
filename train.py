import os 
import time
import numpy as np 
from pathlib import Path 
from datetime import datetime
from utils import time_string

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from tune_sklearn import TuneGridSearchCV

data_file = Path("./data/")
save_file = Path("./report/")

n_cpus = os.cpu_count()

def main(save_route):

    start = time.time()
    
    print("loading dataset...")
    
    X_train = np.load(list(data_file.glob("X_train.npy"))[0])
    y_train = np.load(list(data_file.glob("y_train.npy"))[0])

    X_test = np.load(list(data_file.glob("X_test.npy"))[0])
    y_test = np.load(list(data_file.glob("y_test.npy"))[0])
    
    print(y_test.mean(), y_train.mean())
    
    print(f"done! ({time_string(start)})")

    start = time.time()
    print("Transforming dataset...")
    
    transformer = MaxAbsScaler().fit(X_train)

    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    
    print(f"done! ({time_string(start)})")
    
    start = time.time()
    clf = SGDClassifier(
              loss = "log_loss",
              random_state=42,
              class_weight="balanced",
              shuffle=True,
              penalty='l1',
              )

    # parameters = {
    #     'loss':['log_loss']
    # }

    # tune_search = TuneGridSearchCV(
    #     clf,
    #     parameters,
    #     scoring = 'roc_auc',
    #     cv = None,
    #     verbose = 1,
    #     early_stopping = True,
    #     refit = False,
    #     n_jobs = (n_cpus//2),
    #     local_dir = save_route,
    # )

    print("Fitting tune search...")
    
    clf.fit(X_train, y_train)
        
    print("done! ({time_string(start)})")
    
    print("Registering results...")
        
    # tune_search_res = tune_search.cv_results_
    # best_params = tune_search.best_params_
    
    pred = clf.predict(X_test)    
    score = roc_auc_score(y_test, pred)
    
    info = f"Best ROC-AUC score: {score:.2f}"
    
    print(info)
    
    file_name = datetime.today().strftime('%H:%M:%S')
    
    with open(save_route / file_name, "a") as f:
        f.write(info)
    
    print("done! ({time_string(start)})")
    
    return 


if __name__ == "__main__":
    
    cur_time = datetime.today().strftime('%Y-%m-%d')
    save_route = save_file / f"check_point_{cur_time}"
    
    save_route.mkdir(parents=True,exist_ok=True)
    
    main(save_route)

    



