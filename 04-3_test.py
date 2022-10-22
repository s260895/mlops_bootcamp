import pandas as pd

predict = __import__('04-2_predict')

test_data_path = 'data/green_tripdata_2022-01.parquet'
raw_data = pd.read_parquet(test_data_path)

prediction, score = predict.predict(raw_data.tail(1))
print(prediction, score)