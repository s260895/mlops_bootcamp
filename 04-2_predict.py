
import mlflow
import pandas as pd
train = __import__('04-1_train')
from sklearn.metrics import mean_squared_error
mlflow.set_tracking_uri("http://localhost:5000")

# define function for prediction using saved model
def predict(raw_data):
    # Load model from mlflow as a PyFuncModel.
    logged_model = 'runs:/6da351c6566a42849818c90a5d3aa22d/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    X,y = train.cleaned_train_and_target(raw_data,clean=False)
    # X = loaded_model.transform(X)
    y_pred = loaded_model.predict(X)
    rmse = mean_squared_error(y,y_pred,squared=False)
    return y_pred,rmse

