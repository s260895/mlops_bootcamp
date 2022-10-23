import pandas as pd
pd.options.mode.chained_assignment = None 
import requests
import json
predict = __import__('04-3_predict')

test_payload = {'PULocationID':['12'],
                'DOLocationID':['13'],
                'trip_distance':[12.0],
                'total_amount':[20.0],
                'passenger_count':[1.0]}

url = 'http://localhost:9696/predict'
response = requests.post(url,json=test_payload)
print(response.json())