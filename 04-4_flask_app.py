from flask import Flask, request, jsonify
predict = __import__('04-2_predict')


app = Flask('duration-prediction')

@app.route('/predict',methods=['POST'])
def prediction_endpoint():
    data = request.get_json()
    # print(data)
    prediction = predict.predict(data,train_feat=['PU_DO_pair','trip_distance','total_amount','passenger_count'],target_feat='duration')
    result = {
        'duration': prediction[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
    