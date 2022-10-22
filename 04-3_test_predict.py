import pandas as pd
pd.options.mode.chained_assignment = None 

predict = __import__('04-2_predict')

test_data_path = 'data/green_tripdata_2022-01.parquet'
raw_data = pd.read_parquet(test_data_path)

prediction = predict.predict(raw_data.tail(1),train_feat=['PU_DO_pair','trip_distance','total_amount','passenger_count'],target_feat='duration')
print(prediction)