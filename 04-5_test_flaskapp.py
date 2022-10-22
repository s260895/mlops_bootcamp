import pandas as pd
import requests
import json

pd.options.mode.chained_assignment = None 

predict = __import__('04-2_predict')

test_data_path = 'data/green_tripdata_2022-01.parquet'
raw_data = pd.read_parquet(test_data_path)

test_payload = raw_data.head(1).to_dict(orient='list')
train_feat=['PULocationID','DOLocationID','trip_distance','total_amount','passenger_count']
test_payload = {key:test_payload[key] for key in train_feat}
url = 'http://localhost:9696/predict'
# print(test_payload)
# print([type(elem[0]) for elem in test_payload.values()])
response = requests.post(url,json=test_payload)
print(response.json())