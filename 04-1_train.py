# %%
# data manipulation and storage
import pandas as pd
import numpy as np

# plotting and graphs
import seaborn as sns
import matplotlib.pyplot as plt

# data preprocessing
# from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# models
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

# model performance metrics
from sklearn.metrics import mean_squared_error

# saving model to file
import pickle

# mlflow for experiment tracking
import mlflow

# hyper-parameter optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# sklearn pipeline creation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# misc utilities
import copy

# intel sklearn optimization library
# from sklearnex import patch_sklearn
# patch_sklearn()


# %%
def cleaned_train_and_target(df,train_feat,target_feat,inference=False):

        
    if inference == False:
        # create concatenated categorical feature
        df.loc[:,'PU_DO_pair'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
        # create target feature
        df.loc[:,'duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
        df.loc[:,'duration'] = df['duration'].apply(lambda td: td.total_seconds()/60)
        # filter out rows based on various conditions
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
        df = df[(df['trip_distance'] > 1)&(df['trip_distance'] < 25)]
        df = df[(df['total_amount'] > 1)&(df['total_amount'] < 150)]
        df = df[df['passenger_count'] > 0]  
        y = df[target_feat]
        X = df[train_feat]
    if inference == True:
        df['PU_DO_pair'] = str(df['PULocationID']) + '_' + str(df['DOLocationID'])             
        y = None
        del df['PULocationID']
        del df['DOLocationID']
        X = df
    
    return X,y


# %%
def initialize_regression_model(params,categoric_feat,numeric_feat,type='gradientbooster'):

    categorical = categoric_feat
    numerical = numeric_feat

    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])


    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numerical),
        ('categoric', categorical_pipeline, categorical)
    ])

    if type == 'gradientbooster':
        regressor = GradientBoostingRegressor(**params)
    else:
        regressor = None

    regression_model = Pipeline(steps=[
        ('preprocess', preprocessor_pipeline),
        ('model', regressor)
    ])

    return regression_model


# %%
def hyperparameter_optimizer(
    X_train,
    y_train,
    X_val,
    y_val,
    categoric_feat,
    numeric_feat
):

    # define hyper-parameter search space
    search_space = {
        # 'n_estimators':hp.choice('n_estimators',np.arange(10,101,1)),
        'learning_rate':hp.loguniform('learning_rate',-3,1),
        # 'min_samples_split':hp.loguniform('min_child_weight',-4,0),
        # 'max_depth':scope.int(hp.quniform('max_depth',5,100,5)),        
        'random_state':42
    }

    # define objective function
    def objective(params):
        
        with mlflow.start_run():
            mlflow.set_tag('model','gradientboostingregressor')
            # mlflow.log_params(params)
            mlflow.sklearn.autolog()
            pipe = initialize_regression_model(params=params,categoric_feat=categoric_feat,numeric_feat=numeric_feat,type='gradientbooster')
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_val)
            rmse = mean_squared_error(y_val,y_pred,squared=False)
            mlflow.log_metric('validation_rmse',rmse)
            # mlflow.log_artifact(scaler_path,artifact_path="preprocessor")
            # mlflow.log_artifact(vectorizer_path,artifact_path="preprocessor")
            # mlflow.xgboost.log_model(xgb_model,artifact_path="models_mlflow")
        

        return {'loss':rmse,'status':STATUS_OK}


    # Perform hyper-parameter optimization
    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals=50,
        trials = Trials()
    )   

    return 

# %%
def main(
    train_path = 'data/green_tripdata_2021-01.parquet',
    val_path = 'data/green_tripdata_2021-01.parquet',
    tracking_uri = 'sqlite:///mlflow.db',
    experiment = 'gradient-booster-experiment-2',
    target_feat = 'duration',
    train_feat = ['PU_DO_pair','trip_distance','total_amount','passenger_count'],
    categoric_feat = ['PU_DO_pair'],
    numeric_feat = ['trip_distance','total_amount','passenger_count']
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    train_path = '/home/ub/Documents/git/mlops_bootcamp/data/green_tripdata_2021-01.parquet'
    val_path = '/home/ub/Documents/git/mlops_bootcamp/data/green_tripdata_2021-02.parquet'
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    X_train,y_train = cleaned_train_and_target(df_train,train_feat,target_feat)
    X_val, y_val = cleaned_train_and_target(df_val,train_feat,target_feat)
    hyperparameter_optimizer(X_train,y_train,X_val,y_val,categoric_feat,numeric_feat)
    # train_best_model(train,valid,y_val,dv,scaler)

# main()

# %%



