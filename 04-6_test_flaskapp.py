import pandas as pd

import requests
import json

test_payload = {'PULocationID':['12'],
                'DOLocationID':['13'],
                'trip_distance':[12.0],
                'total_amount':[20.0],
                'passenger_count':[1.0]}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url,json=test_payload)
print(response.json())