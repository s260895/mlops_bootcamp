import pandas as pd
data_clean = __import__('04-1_clean')

# define function for prediction using saved model
def predict(raw_data,train_feat,target_feat,model):
    
    # Load model from mlflow as a PyFuncModel.
    X,y = data_clean.cleaned_train_and_target(df=pd.DataFrame(raw_data),train_feat=train_feat,target_feat=target_feat,inference=True)
    # X = loaded_model.transform(X)
    y_pred = model.predict(X)
    # rmse = mean_squared_error(y,y_pred,squared=False)
    return y_pred
