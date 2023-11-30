import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# data visualization
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt

# machine learning

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


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

def plot_stats(df,col):
    dataframe = df[['Date',col]]
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe.set_index('Date',inplace=True)
    print(dataframe.head())
    dataframe = dataframe.resample('W').mean()
    fig = px.line(dataframe,title=f'Time Series Plot of {col}',)
    fig.update_layout(
    xaxis_title='Date', yaxis_title=col,).update_traces(line_color='tomato')
    return fig

def plot_sales(data,col,x='Month',y='Sales',color=None):
    if color:
        new_df = data.groupby([x,col,color])[y].mean().reset_index()
        fig = px.line(new_df,x=x,y=y,facet_col=col,color=color,markers=True,title=f'Sales Comparison: {col} vs No {col}')#.update_traces(line_color='red')
        fig.update_layout(
        autosize=True,width=800,height=400,
        yaxis_title='Sales',showlegend=True)
    else:
        new_df = data.groupby([x,col])[y].mean().reset_index()
        fig = px.line(new_df,x=x,y=y,facet_col=col,markers=True,title=f'Sales Comparison: {col} vs No {col}').update_traces(line_color='red')
        fig.update_layout(
        autosize=False,width=800,height=400,
        yaxis_title='Sales',showlegend=False)
    fig.update_xaxes(tickvals=new_df[x])
    
    return fig


#Factor plot
def plot_cat(data,x,y,col,hue):
    sns.catplot(data=data,x=x,y=y,col=col,hue=hue,kind='point')
    plt.show()
#Bar plot    
def plot_bar(data,x,y1,y2):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.barplot(data = data, x = x, y = y1,palette='RdYlBu') 
    plt.title(f'{x} VS {y1}')
    plt.subplot(1,2,2)
    sns.barplot(data = data, x = x, y = y2,palette='RdYlGn')
    plt.title(f'{x} VS {y2}')
    plt.show()
#Histogram
def plot_hist(data,col1,col2):
    sns.histplot(data=data,x=col1,y=col2,bins=30)
    plt.title(f'Distribution of {col1} by {col2}')
    plt.show()