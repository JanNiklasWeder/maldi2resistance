import sys
sys.path.append('../')
import copy
import joblib
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics.classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from maldi2resistance.metric.ROC import MultiLabelRocNan
from maldi2resistance.metric.PrecisionRecall import MultiLabelPRNan
from maldi2resistance.data.driams import Driams


driams = Driams(
    root_dir="/scratch1/users/park11/driams",
    sites=["DRIAMS-B"],
)
driams.loading_type = "memory"

def get_model_pipeline(model:str, random_state, class_weight='balanced'):
    if model == 'lr':                                                                                                                                                                                                                 
        lr = LogisticRegression(
                solver='saga',
                max_iter=500,
                class_weight=class_weight,
                random_state=random_state
        )

        pipeline = Pipeline(
            steps=[
                ('scaler', None),
                ('lr',MultiOutputClassifier(lr, n_jobs=-1)),
            ]
        )

        param_grid = [
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__estimator__C': 10.0 ** np.arange(-2, -1),
                'lr__estimator__penalty': ['l1', 'l2'],
            },
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__estimator__C': 10.0 ** np.arange(-2, -1),
                'lr__estimator__penalty': ['elasticnet'],
                'lr__estimator__l1_ratio': [0.5],
            }
        ]

        return pipeline, param_grid

    elif model == 'rf':
        # Make sure that we set a random state here; else, the results
        # are not reproducible.
        if random_state is None:
            warnings.warn(
                '`random_state` is not set for random '
                'forest classifier.'
            )

        rf = RandomForestClassifier(
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )

        pipeline = Pipeline(
            steps=[
                ('rf', MultiOutputClassifier(rf, n_jobs=-1)),
            ]
        )

        param_grid = [ 
            {
                'rf__estimator__criterion': ['gini', 'entropy'],
                'rf__estimator__bootstrap': [False],
                'rf__estimator__n_estimators': [100, 200, 400],
                'rf__estimator__max_features': ['auto', 'sqrt', 'log2']
            },
            {
                'rf__estimator__criterion': ['gini', 'entropy'],
                'rf__estimator__bootstrap': [True],
                'rf__estimator__oob_score': [True, False],
                'rf__estimator__n_estimators': [100, 200, 400],
                'rf__estimator__max_features': ['auto', 'sqrt', 'log2']
            },
        ]

        return pipeline, param_grid

    elif model == 'knn':
        # Make sure that we set a random state here; else, the results
        # are not reproducible.
        if random_state is None:
            warnings.warn(
                '`random_state` is not set for random '
                'forest classifier.'
            )

        neigh = KNeighborsClassifier(
            n_jobs=-1,
            n_neighbors=5,
            weights="distance", p=1
        )    

        pipeline = Pipeline(
            steps=[
                ('knn', MultiOutputClassifier(neigh, n_jobs=-1)),
            ]
        )

        param_grid = [ 
            {
                'knn__estimator__n_neighbors': [3,5,10],
                'knn__estimator__weights': ['uniform', 'distance'],
                'knn__estimator__p': [1, 1.5, 2],
                'knn__estimator__leaf_size': [15, 30, 60],
            },
        ]

        return pipeline, param_grid


def nan_roc_cal(y, y_pred, micro=True, **kwargs):
    output = torch.Tensor(np.array(y_pred))
    test_labels = torch.Tensor(np.array(y))
    ml_roc = MultiLabelRocNan()
    micro, macro = ml_roc.compute(output,test_labels,driams.selected_antibiotics)
    if micro:
        return micro
    else:
        return macro


def run(
        X_train, y_train,
        X_test, y_test,
        model,
        n_folds=5,
        random_state=None,
        class_weight='balanced'
    ):
    pipeline, param_grid = get_model_pipeline(
        model,
        random_state,
        class_weight=class_weight,
    )

    nan_roc = make_scorer(nan_roc_cal, micro=True, greater_is_better=True)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=n_folds,
        scoring=nan_roc,
        n_jobs=-1,
    )

    with warnings.catch_warnings():                                                                                                                                                                                                   
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        with joblib.parallel_backend('threading', -1):
            grid_search.fit(X_train, y_train)

    train_metrics = calculate_metrics(
        y_train,
        grid_search.predict(X_train),
        grid_search.predict_proba(X_train),
        prefix='train'
    )

    y_pred = grid_search.predict(X_test)
    y_score = grid_search.predict_proba(X_test)

    test_metrics = nan_roc(y_test, y_pred)
    print('\n')
    print(test_metrics)

    output_filename = f'./results_ml-sklearn/Model_{model}_Seed_{random_state}_Data_Driams'
    ml_roc = MultiLabelRocNan()
    print(ml_roc.compute(y_pred,y_score,driams.selected_antibiotics, create_csv=f"{output_filename}_ROC.csv"))
    fig_, ax_ = ml_roc()
    plt.savefig(f"{output_filename}_ROC.png", transparent=True, format= "png", bbox_inches = "tight")
    plt.close()

    ml_pr = MultiLabelPRNan()
    print(ml_pr.compute(y_pred,y_score,driams.selected_antibiotics, create_csv=f"{output_filename}_PR.csv"))
    fig_, ax_ = ml_pr()
    plt.savefig(f"{output_filename}_PR.png", transparent=True, format= "png", bbox_inches = "tight")
    plt.close()

if __name__ == '__main__':
    SEED = int(sys.argv[2])
    torch.manual_seed(SEED)
    
    gen = torch.Generator()
    gen.manual_seed(SEED)

    train_size = int(0.8 * len(driams))
    test_size = len(driams) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(driams, [train_size, test_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=True)

    train_dataset_array = next(iter(train_loader))[0].numpy()
    train_dataset_label = next(iter(train_loader))[1].numpy()
    np.nan_to_num(train_dataset_label, copy=False, nan=0.0)
    train_dataset_label = pd.DataFrame(train_dataset_label.tolist())

    print(train_dataset_label.sum(axis=1))

    test_dataset_array = next(iter(test_loader))[0].numpy()
    test_dataset_label = next(iter(test_loader))[1].numpy()
    np.nan_to_num(test_dataset_label, copy=False, nan=0.0)
    test_dataset_label = pd.DataFrame(test_dataset_label.tolist())

    print(test_dataset_label.sum(axis=1))

    print(f'run model: {sys.argv[1]} random_state: {sys.argv[2]}')
    # Run grid search
    run(
        train_dataset_array, train_dataset_label,
        test_dataset_array, test_dataset_label,
        model=sys.argv[1],
        random_state=SEED,
    )
