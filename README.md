# HipFracT2D

The project is intend to build binary classifier on Hip Fracture outcome from patients diagnosed with Type-2 diabetes. The data source is MarketScan, we have created some toy samples under data directory for testing purpose. This repo contains the complete scripts for train and inference. 

The train script include two types of model, stochastic-gradient-descent (sgd) linear module and gradient boosting (xgb) module. Both samples are built with tunable hyper-parameters via setup file under train directory. 

The xgboost model under model directory is currently the best well tunned model shows state-of-the-art performance. 

![alt text](https://github.com/chengjianshi/HipFracT2D/blob/main/figure/best_mdoel_train_cm_roc.png "train confusion matrix")
![alt text](https://github.com/chengjianshi/HipFracT2D/blob/main/figure/best_model_test_cm_roc.png "test confusion matrix")

#### environment setup 

```bash
python -V 
Python 3.8.5

conda install --file requirements.txt
```

Train script 

```bash
python train/train_xgb.py 
```

Infer script 

```bash

python infer.py model/ data/ 

```

