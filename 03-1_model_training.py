# data manipulation and storage
import pandas as pd

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


class DataFrameCleaner(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self):
        # save the features list internally in the class
        self.y = pd.DataFrame()
        return
        
    def fit(self, X, y = None):
        return self    
        
    def transform(self, X, y=None):
        # return the dataframe with the specified features
        self.X = X
        self.y['duration'] = self.X['lpep_dropoff_datetime'] - self.X['lpep_pickup_datetime']
        self.y['duration'] = self.y['duration'].apply(lambda td: td.total_seconds()/60)
        self.X['duration'] = self.X[(self.X['duration'] >= 1) & (self.X['duration'] <= 60)]
        self.X = self.X[(self.X['trip_distance'] > 1)&(self.X['trip_distance'] < 25)]
        self.X = self.X[(self.X['total_amount'] > 1)&(self.X['total_amount'] < 150)]
        self.X = self.X[self.X['passenger_count'] > 0]  
        self.X['PU_DO_pair'] = self.X['PULocationID'].astype(str) + '_' + self.X['DOLocationID'].astype(str)                
        return self.X, self.y

data_cleaning_pipeline = Pipeline(steps=[('clean', DataFrameCleaner())])
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor_pipeline = ColumnTransformer(transformers=[
    ('clean', data_cleaning_pipeline, numerical_features+categorical_features),
    ('numeric', numeric_pipeline, numerical_features),
    ('categoric', categorical_pipeline, categorical_features)
])

regression_model = GradientBoostingRegressor()

regressor_pipe = Pipeline(steps=[
    ('preprocess', preprocessor_pipeline),
    ('model', regression_model)
])




# # define function to clean dataset
# def read_and_clean_dataframe(
#     df
# ):
#     df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
#     df['duration'] = df.duration.apply(lambda td: td.total_seconds()/60)
#     df = df[(df.duration >= 1) & (df.duration <= 60)]
#     df = df[(df.trip_distance > 1)&(df.trip_distance < 25)]
#     df = df[(df.total_amount > 1)&(df.total_amount < 150)]
#     df = df[df['passenger_count'] != 0]  
#     df['PU_DO_pair'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
#     return df

def preprocess_dataframe(
    train_path,
    val_path
):
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_train = read_and_clean_dataframe(df_train)
    df_val = read_and_clean_dataframe(df_val)
    print(len(df_train))
    print(len(df_val))

    categorical = ['PU_DO_pair']
    # ['PULocationID','DOLocationID']
    numerical = ['trip_distance','fare_amount']
    target = 'duration'
    # Pre Processing - Numerical
    scaler = StandardScaler()
    df_train[numerical] = scaler.fit_transform(df_train[numerical])
    df_val[numerical] = scaler.transform(df_val[numerical])
    train_dicts = df_train[categorical+numerical].to_dict(orient='records')
    val_dicts = df_val[categorical+numerical].to_dict(orient='records')
    # Pre Processing - Categorical
    df_train[categorical] = df_train[categorical].astype(str)
    df_val[categorical] = df_val[categorical].astype(str)
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train[target].values
    X_val = dv.transform(val_dicts)
    y_val = df_val[target].values

    vectorizer_path = 'models/vectorizer.b'
    scaler_path = 'models/scaler.b'

    with open(vectorizer_path,'wb') as f_out:
        pickle.dump(dv,f_out)
    
    with open(scaler_path,'wb') as f_out:
        pickle.dump(scaler,f_out)

    return X_train,y_train,X_val,y_val, scaler_path, scaler_path

def hyperparameter_optimizer(
    X_train,
    y_train,
    X_val,
    y_val,
    scaler_path,
    vectorizer_path
):

    # define hyper-parameter search space
    search_space = {
        'max_depth':scope.int(hp.quniform('max_depth',4,100,1)),
        'learning_rate':hp.loguniform('learning_rate',-3,0),
        'reg_alpha':hp.loguniform('reg_alpha',-5,-1),
        'reg_lambda':hp.loguniform('reg_lambda',-6,-1),
        'min_child_weight':hp.loguniform('min_child_weight',-1,3),
        'objective':'reg:squarederror',
        'seed':42
    }

    train = xgb.DMatrix(X_train,label=y_train)
    valid = xgb.DMatrix(X_val, label = y_val)

    # define objective function
    def objective(params):
        with mlflow.start_run():
            # mlflow.set_tag('model','xgboost')
            # mlflow.log_params(params)
            mlflow.xgboost.autolog()
            xgb_model = xgb.train(
                params = params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid,'validation')],
                early_stopping_rounds=50
            )
            y_pred = xgb_model.predict(valid)
            rmse = mean_squared_error(y_val,y_pred,squared=False)
            # mlflow.log_metric('rmse',rmse)

            mlflow.log_artifact(scaler_path,artifact_path="preprocessor")
            mlflow.log_artifact(vectorizer_path,artifact_path="preprocessor")
            # mlflow.xgboost.log_model(xgb_model,artifact_path="models_mlflow")
        

        return {'loss':rmse,'status':STATUS_OK}


    # Perform hyper-parameter optimization
    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals=2,
        trials = Trials()
    )   

    return 


# def train_best_model(
#     train,
#     valid,
#     y_val,
#     dv,
#     scaler
# ):
#     # Perform single trial
#     # with Auto Logging for MLFlow enabled

#     with mlflow.start_run():

#         params = {
#             'learning_rate'	:0.1700079110741563,
#             'max_depth':	16,
#             'min_child_weight':2.37717271395477,
#             'objective':	'reg:squarederror',
#             'reg_alpha':	0.363258077609188,
#             'reg_lambda':	0.011733914718256189,
#             'seed':	42
#         }

#         mlflow.log_params(params)
#         mlflow.xgboost.autolog()

#         booster = xgb.train(
#             params = params,
#             dtrain = train,
#             num_boost_round = 1000,
#             evals = [(valid,'validation')],
#             early_stopping_rounds = 50
#         )

#         y_pred = booster.predict(valid)
#         rmse = mean_squared_error(y_val,y_pred,squared=False)
#         mlflow.log_metric('rmse',rmse)

#         mlflow.log_artifact('models/vectorizer.b',artifact_path="preprocessor")
#         mlflow.log_artifact('models/scaler.b',artifact_path="preprocessor")
#         mlflow.xgboost.log_model(booster,artifact_path="models_mlflow")
    
#     return


def main(
    train_path = 'data/green_tripdata_2021-01.parquet',
    val_path = 'data/green_tripdata_2021-01.parquet',
    tracking_uri = 'sqlite:///mlflow.db',
    experiment = 'test-experiment'
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    X_train,y_train,X_val,y_val, dv_path, scaler_path = preprocess_dataframe(train_path,val_path)
  
    hyperparameter_optimizer(train,valid,y_val)
    # train_best_model(train,valid,y_val,dv,scaler)

main()