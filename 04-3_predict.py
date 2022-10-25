from bleach import clean
import mlflow
import pandas as pd
clean = __import__('04-1_clean')
from sklearn.metrics import mean_squared_error
mlflow.set_tracking_uri("http://localhost:5000")

# define function for prediction using saved model
def predict(raw_data,train_feat,target_feat):
    # Load model from mlflow as a PyFuncModel.
    logged_model = 'runs:/9a28033fdd914d0eb4ae48fb11fde866/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    X,y = clean.cleaned_train_and_target(df=pd.DataFrame(raw_data),train_feat=train_feat,target_feat=target_feat,inference=True)
    # X = loaded_model.transform(X)
    y_pred = loaded_model.predict(X)
    # rmse = mean_squared_error(y,y_pred,squared=False)
    return y_pred

