import os 
import sys 
import time
import joblib
import numpy as np 
from scipy import sparse
from pathlib import Path 
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(12345)
sys.path.append("/home/chengjian/scratch-midway2/HipFracT2D/")

from utils import time_string, load_parameters

data_file = Path("/home/chengjian/scratch-midway2/HipFracT2D/data/")
report_file = Path("/home/chengjian/scratch-midway2/HipFracT2D/report/")
params_file = Path("/home/chengjian/scratch-midway2/HipFracT2D/train/setup/sgd_params.txt")

n_cpus = os.cpu_count()

print(f"Numebr of cpus available: {n_cpus} \n")

def main(save_route: Path):

# -----------------------------------------------------------------------------
    
    print(f"Initializing SGDClassifier training, save_path: {save_route.name} \n")

    tot_start = time.time()
    start = time.time()
    
# -----------------------------------------------------------------------------

    print("Loading dataset...")
    
    X_train = sparse.load_npz(data_file / "npz_data/X_train_red.npz").toarray()
    y_train = sparse.load_npz(data_file / "npz_data/y_train.npz").toarray().squeeze()
    
    X_test = sparse.load_npz(data_file / "npz_data/X_test_red.npz").toarray()
    y_test = sparse.load_npz(data_file / "npz_data/y_test.npz").toarray().squeeze()

    print(f"done! ({time_string(start)}) \n")

    print("Training...")
    
    start = time.time()
    
    params = load_parameters(params_file)
    clf = SGDClassifier(**params)
    
    clf.fit(X_train, y_train)
    
    print(f"done! ({time_string(start)}) \n")

# -----------------------------------------------------------------------------

    print("Registering results...")
    start = time.time()
    
    y_test_pred = clf.predict(X_test)    
    y_train_pred = clf.predict(X_train)
    
    test_score = roc_auc_score(y_test, y_test_pred)
    train_score = roc_auc_score(y_train, y_train_pred)
    
    info = f''' 
    {'*' * 30} 
    Train ROC-AUC score: {train_score:.5f}
    Test ROC-AUC score: {test_score:.5f}
    Total time: {time_string(tot_start)}
    save route: {save_route} 
    {'*' * 30} \n
    '''
    
    joblib.dump(clf, save_route / f"model.pkl")
    
    with open(save_route / "log.txt", "w") as f:
        f.write(info)
        for k, v in params.items():
            f.write(str(k) + ':'+ str(v) + '\n')
    
    print(info)

    return 


if __name__ == "__main__":
    
    cur_time = datetime.today().strftime('%Y-%m-%d-%H-%M')
    save_route = report_file / f"check_point_{cur_time}"
    
    save_route.mkdir(parents=True,exist_ok=True)
    
    main(save_route)

    



