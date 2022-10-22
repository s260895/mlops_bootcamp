import mlflow
import pandas as pd
import 04-1_train


# Load model from mlflow as a PyFuncModel.
logged_model = 'runs:/6da351c6566a42849818c90a5d3aa22d/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# define function for prediction using saved model
def predict(df):
    X = loaded_model.transform(df)
    y_pred = loaded_model.predict(X)