import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
#import plotting and ml functions
from functions import *
#import awesome_streamlit as ast


st.set_page_config(page_title="Rossmann Pharmaceuticals Store Sales", layout="wide")

st.title('Rossman Pharmaceuticals Store Sales Analysis')
st.write("""
         The goal of this [project](https://github.com/katenjoki/rossman_sales_prediction) is to:\n
         * extract insights from the dataset and 
         * forecast sales in all Rossman Pharmaceutical stores across several cities six weeks ahead of time.
         """)

st.write("""
         This dashboard has 2 sections that can be seen on the sidebar to the left. \n
         * Plot features (current) - This section contains different feature visualizations. 
            * On the sidebar, select features to visualize.
         * Regression Models - This section runs different regression models and compares their individual evaluation metrics.
         """)

with st.expander("Most of the fields are self-explanatory. Click here to get a quick description of some features:"):
    st.write("""
    * Open - an indicator for whether the store was open: 0 = closed, 1 = open

    * StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. 
    Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

    * SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

    * StoreType - differentiates between 4 different store models: a, b, c, d

    * Assortment - describes an assortment level: a = basic, b = extra, c = extended.
             
    * CompetitionDistance - distance in meters to the nearest competitor store

    * CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

    * Promo - indicates whether a store is running a promo on that day

    * Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
    """)
st.sidebar.text('Exploratory Data Analysis!')
feature_name = st.sidebar.selectbox(
    "Select features to visualize",(
        "Correlation","Time Series Plot of Sales","Promotions","Store Type","Assortment","Competition Distance"))

clean_train = pd.read_csv("clean_train.csv",compression='gzip',)
train_store = pd.read_csv("train_store.csv",compression='gzip',)
test_store = pd.read_csv("test_store.csv")

#st.table(clean_train.head())
#Factor plot
def plot_factor(data,x,y,col,hue):
    sns.factorplot(data=data,x=x,y=y,col=col,hue=hue)
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


if feature_name == "Time Series Plot of Sales":
    st.plotly_chart(plot_stats(clean_train,'Sales'))
    st.write("""
    - On average, sales are lowest days after Christmas and 31st December and highest a few weeks to Christmas
    - Because most stores close on Christmas, a few days before the holiday, sales significantly increase and start dropping from 23rd
    - We can also observe a mid-year peak, at around March to May. 
             """)
elif feature_name == "Correlation":
    st.subheader("Correlation of features in the Rossmann Store Sales Dataset")
    fig = plt.figure(figsize = (8,6))
    cols = clean_train.select_dtypes(exclude='object').columns.tolist()
    print('-------------------------------')
    print(cols)
    cols.remove('Store')
    cols.remove('Sales_per_Customer')
    fig = px.imshow(clean_train[cols].corr(),color_continuous_scale='reds',title='Correlation Matrix')
    fig.update_layout( autosize=False,width=800,height=800,)
    st.plotly_chart(fig)
    st.divider()
    fig = px.scatter(clean_train,x='Customers',y='Sales',title='Sales vs Customers',trendline='ols',trendline_color_override='red')
    st.plotly_chart(fig)
    st.write("""
    - Sales and Customers have a strong positive correlation of ~ 0.8
    - Sales and Promo also have a moderate positive correlation of ~ 0.4""")
elif feature_name == 'Promotions':
    st.subheader("Promo vs Promo2")
    st.write("""
        * Promo - indicates whether a store is running a promo on that day. 
            * 0 = No promo, 1 = Promo
        * Promo2 - Promo2 is a continuing and consecutive promotion for some stores.
            * 0 = store is not participating, 1 = store is participating
            """)
    st.plotly_chart(plot_sales(train_store,'Promo'))
    st.divider()
    st.plotly_chart(plot_sales(train_store,'Promo2'))
    st.write("""
        - There are significantly higher sales, overall, when stores have a Promo
        - However,stores participating in Promo2 seem to have **LOWER SALES**, than those not participating.
        - In fact, we have more stores participating in Promo2 yet they have lower sales!
        """)
    st.table(train_store.groupby('Promo2').agg({'Store':'nunique','Sales':'sum'}).reset_index().rename(columns={'Sales':'Total Sales'}))
elif feature_name == "Store Type":
    st.subheader("Sales Comparison: Store Type")
    st.plotly_chart(plot_sales(train_store,'Promo',color='StoreType'))
    st.divider()
    st.plotly_chart(plot_sales(train_store,'SchoolHoliday',color='StoreType'))
    st.divider()
    st.plotly_chart(plot_sales(train_store,'SchoolHoliday',color='StoreType'))
    st.table(train_store.groupby(['StoreType']).agg({'Store':['nunique','count'],'Sales':['sum','mean'],'Customers':'sum'}).reset_index())
    st.write("""
    * StoreType b has the lowest number of stores but highest average sales, regardless of regardles of Promo, State Holiday or School Holiday. 
    * Stores under in StoreType b could possible sell low priced, frequently bought items.
    * However, we can see that promotions lead to higher sales for all store types""")
elif feature_name == "Assortment":
    st.subheader("Sales Comparison: Assortment")
    st.write("""
    * Assortment - describes an assortment level: a = basic, b = extra, c = extended""")
    st.plotly_chart(plot_sales(train_store,'Promo',color="Assortment"))
    st.divider()
    st.plotly_chart(plot_sales(train_store,'SchoolHoliday',color="Assortment"))
    st.divider()
    st.plotly_chart(plot_sales(train_store,'SchoolHoliday',color="Assortment"))
    st.write("""
    * Promotions lead to higher sales for Assortments a and c.
    * Assortment b has higher sales without a promotion, than with a promotion.
    * Assortment c has higher sales than b and c, regardles of whether there's a School Holiday/ State Holiday or not.""")

else:
    st.subheader("Competition Distance")
    fig = px.scatter(train_store,x='CompetitionDistance',y='Sales',title='Sales vs Competition Distance (metres)')
    st.plotly_chart(fig)
    fig = plt.figure(figsize = (8,6))
    #sns.set_theme(style='dark')
    #sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
    s = sns.distplot(train_store.CompetitionDistance, color = 'red',)
    #s.set_facecolor('#000000')
    plt.suptitle('Distribution of Competition Distance in metres',)#y=1.02)
    st.pyplot(fig)
    st.write(s)
    st.write("""
    * The stores could be located in densely populated areas hence, distance to nearest competitor has a small influence.
    * Most stores are located around 5km from competitors""")

