import os
import mlflow

from flask import Flask, request, jsonify
predict = __import__('04-3_predict')

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)
logged_model = 'runs:/9a28033fdd914d0eb4ae48fb11fde866/model/MLmodel'
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = Flask('duration-prediction')

@app.route('/predict',methods=['POST'])
def prediction_endpoint():
    data = request.get_json()
    # print(data)
    prediction = predict.predict(data,train_feat=['PU_DO_pair','trip_distance','total_amount','passenger_count'],target_feat='duration',model=loaded_model)
    result = {
        'duration': prediction[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
    