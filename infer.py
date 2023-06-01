import os 
import sys 
import pandas as pd 

from pathlib import Path
from scipy import sparse
from xgboost.sklearn import XGBClassifier
from typing import Union
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt

cwd = os.getcwd()

report_path = Path(f'{cwd}/report')
figure_path = Path(f'{cwd}/figure')

def main(model_path: Union(Path, str), 
         data_path: Union(Path, str)):
    
    X_sparse = sparse.load_npz(data_path / "X_toy_sample.npz")
    y = sparse.load_npz(data_path / "y_toy_sample.npz").toarray().squeeze()

    best_model = XGBClassifier()
    best_model.load_model(model_path)
    y_pred = best_model.predict(X_sparse)
    y_prob_pred = best_model.predict_proba(X_sparse)[:,1]
    
    plot_confusion_matrix(y, y_pred, y_prob_pred, target_names=["HipFrac", "NonHipFrac"], title="test")
    
    features = []
    with open(data_path / "features.txt", "r") as f:
        for line in f:
            features.append(line.strip())

    df_xgb_w = pd.DataFrame.from_dict({f: w for f,w in zip(features,best_model.feature_importances_)}, orient = "index", columns=['weight'])
    df_xgb_w.sort_values(by = "weight", inplace = True, ascending = False)
    df_xgb_w = df_xgb_w.reset_index().rename(columns={'index': 'Disease'})
    
    df_xgb_w.sort_values(by = "weight", ascending=True, inplace = True)

    fig, ax = plt.subplots(figsize=(8, 15))
    df_xgb_w.set_index("Disease").iloc[-30:].plot.barh(y='weight', ax=ax, legend=False)

    # Set plot title and axis labels
    ax.set_title('Ranked XGboost Feature Importance (Top 30)')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')

    # Set tick label size
    ax.tick_params(axis='both', which='major')

    # Adjust plot margins
    plt.subplots_adjust(left=0.25, bottom=0.15)
    
    return 


if __name__ == "__main__":
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    main(model_path, data_path)