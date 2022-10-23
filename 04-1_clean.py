import pandas as pd

# %%
def cleaned_train_and_target(df,train_feat,target_feat,inference=False):

        
    if inference == False:
        # create concatenated categorical feature
        df.loc[:,'PU_DO_pair'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
        # create target feature
        df.loc[:,'duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
        df.loc[:,'duration'] = df['duration'].apply(lambda td: td.total_seconds()/60)
        # filter out rows based on various conditions
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
        df = df[(df['trip_distance'] > 1)&(df['trip_distance'] < 25)]
        df = df[(df['total_amount'] > 1)&(df['total_amount'] < 150)]
        df = df[df['passenger_count'] > 0]  
        y = df[target_feat]
        X = df[train_feat]
    if inference == True:
        df['PU_DO_pair'] = str(df['PULocationID']) + '_' + str(df['DOLocationID'])             
        y = None
        del df['PULocationID']
        del df['DOLocationID']
        X = df
    
    return X,y