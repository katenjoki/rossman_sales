import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#ML OPS
import mlflow
import dvc.api
import logging
from mlflow import sklearn, log_metric, log_param, log_artifacts
from mlflow.models.signature import infer_signature


from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

logger = logging.getLogger(__name__)


path = "ml_data/train_store.csv"
repo = 'C:/Users/user/Desktop/Projects/rossman_sales'
#version = 'v1'
version = 'v2'

data_url_ = dvc.api.get_url(path=path,rev=version)
print('---------------------------------------')
print(data_url_)


mlflow.set_experiment('Rossman_Sales')

def transform_features(data):
    df = data.copy()
    #label encoding ordinal columns
    le = LabelEncoder()
    ord_columns = ['Promo2SinceYear','CompetitionOpenSinceYear','Year']

    ord_df = pd.DataFrame()
    for i in ord_columns:
        ord_df[i] = le.fit_transform(df[i].astype(str))

    #one hot encode nominal columns with few unique variables
    one_cols = ['StateHoliday','StoreType','Assortment','PromoInterval']
    dummies = pd.get_dummies(df[one_cols])

    #list of numerical columns
    num_cols = ['CompetitionDistance']
    #standardise numerical columns
    min_max = MinMaxScaler()

    num_df = df.loc[:,num_cols]
    num_df[num_cols] = min_max.fit_transform(num_df[num_cols])

    other_cols = [col for col in df.columns if col not in ord_columns if col not in one_cols if col not in num_cols]
    other_df = df[other_cols]
    transform_df = pd.concat([other_df.reset_index(drop=True),num_df.reset_index(drop=True),ord_df.reset_index(drop=True),
                      dummies.reset_index(drop=True)],axis=1,ignore_index=True)

    transform_cols = [list(other_cols),list(num_df.columns),list(ord_df.columns),list(dummies.columns)]

    flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]

    transform_df.columns = flatten(transform_cols)
    return transform_df

def loss_function(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r_squared = r2_score(actual,pred)
    print('RMSE:',rmse)
    print('MAE',mae)
    print('R squared',r_squared)
    return rmse,mae,r_squared

def split_data(train:pd.DataFrame,test:pd.DataFrame):
    ml_cols = test.columns.tolist()
    ml_cols.remove('Store')
    X = train[ml_cols]
    y = train[['Sales']]
    train_x,test_x,train_y,test_y= train_test_split(X,y)
    #encode, transform features after split
    print('-----------------------------------------')
    print('transforming features')
    train_x = transform_features(train_x)
    test_x = transform_features(test_x) 
    return train_x,test_x,train_y,test_y

def train_model(train:pd.DataFrame,test:pd.DataFrame,model,model_name):
    X_train,X_test,y_train,y_test = split_data(train,test)
    #log parameters
    with mlflow.start_run():
        mlflow.log_param('data_url',data_url_)
        mlflow.log_param('data_version',version)
        mlflow.log_param('input_rows',train.shape[0])
        mlflow.log_param('input_cols',train.shape[1])
        #Log artifacts: columns used for modeling
        features = pd.DataFrame(list(X_train.columns))
        features.to_csv('features.csv',header=False,index=False)
        mlflow.log_artifact('features.csv')

        targets = pd.DataFrame(list(y_train.columns))
        targets.to_csv('targets.csv',header=False,index=False)
        mlflow.log_artifact('targets.csv')

        #MODEL
        print(f'Fitting {model_name} model')
        model.fit(X_train,y_train)
        model_pred = model.predict(X_test)

        #log the model
        mlflow.log_param("model", model_name)
        print('---------------------------------------')
        print('Generating evaluation metrics')
        rmse,mae,r_squared = loss_function(y_test,model_pred)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r squared",r_squared)
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print('---------------------------------------')
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    #read train store data from the remote repository
    print('reading data from remote repository')
    train_store = pd.read_csv(data_url_)
    test_store = pd.read_csv('data/test_store.csv')
    rand = RandomForestRegressor(random_state=37)
    train_model(train=train_store,test=test_store,model=rand,model_name='RandomForestRegressor')
    dtree = DecisionTreeRegressor(random_state=47)
    train_model(train=train_store,test=test_store,model=dtree,model_name='DecisionTreeRegressor')