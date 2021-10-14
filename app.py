import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import awesome_streamlit as ast


st.set_page_config(page_title="Rossmann Pharmaceuticals Store Sales Database", layout="wide")

st.title('Rossmann Pharmaceuticals Store Sales Prediction')
with st.expander("Most of the fields are self-explanatory, but below is a quick description of some features:"):
    st.write("""
    * Open - an indicator for whether the store was open: 0 = closed, 1 = open

    * StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. 
    Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

    * SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

    * StoreType - differentiates between 4 different store models: a, b, c, d

    * Assortment - describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here

    * CompetitionDistance - distance in meters to the nearest competitor store

    * CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

    * Promo - indicates whether a store is running a promo on that day

    * Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
    """)

feature_name = st.sidebar.selectbox("Select features to visualize",("Correlation","Sales vs Month, given Promo and Store Type",
"Customers vs Month, given Promo and Store Type","Sales per Customer vs Month, given Promo and Store Type",
"Day of Week vs Sales, Day of Week vs Customers","Competition Distance"))

classifier_name = st.sidebar.selectbox("Select the forecasting model",("Random Forest Regressor",
"Decision Tree Regressor","SGD Regressor"))


train_store = pd.read_csv("data/train_store.csv")
clean_data = pd.read_csv("data/clean_data.csv")
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
if feature_name == "Correlation":
    st.subheader("Correlation of features in the Rossmann Store Sales Dataset")
    fig = plt.figure(figsize = (8,6))
    corr=train_store.loc[:,train_store.columns != "Open"].corr()
    s = sns.heatmap(corr, cmap="Reds")
    st.pyplot(fig)
    st.write(s)
    st.write("""
    * Customers and Sales have a strong positive correlation 
    * Sales and Promo also have a strong positive correlation""")
elif feature_name == "Sales vs Month, given Promo and Store Type":
    st.subheader("Sales vs Month, given Promo and Store Type")
    st.pyplot(plot_factor(train_store,'Month',"Sales",'Promo','StoreType'))
    st.write("""
    * Store Type b has the highest sales per month overall, with and without the promo. 
    * However, we can see that promotions lead to higher sales for all store types""")
elif feature_name == "Customers vs Month, given Promo and Store Type":
    st.subheader("Customers vs Month, given Promo and Store Type")
    st.pyplot(plot_factor(train_store,'Month',"Customers",'Promo','StoreType'))
    st.write("""
    * Store Type b has the highest customers per month overall, with and without the promo. 
    * However, we can see that promotions lead to higher customers for all store types""")
elif feature_name == "Sales per Customer vs Month, given Promo and Store Type":
    st.subheader("Sales per Customer vs Month, given Promo and Store Type")
    st.pyplot(plot_factor(train_store,'Month','Sales_per_Customer','Promo','StoreType'))
    st.write("""
    * StoreType b  has the lowest sales_per_customer, even though it has the highest sales and customers in general.
    * This means that the store gets lots of customers who by many low-value goods.""")
elif feature_name == "Day of Week vs Sales, Day of Week vs Customers":
    st.subheader("Day of Week vs Sales, Day of Week vs Customers")
    st.pyplot(plot_bar(train_store,'DayOfWeek','Sales','Customers'))
    st.write("""
    * Mondays and Sundays have the most sales and customers.
    * Sundays have the most customers""")
else:
    st.subheader("Competition Distance")
    fig = plt.figure(figsize = (8,6))
    s = sns.distplot(train_store.CompetitionDistance, color = 'purple')
    st.pyplot(fig)
    st.write(s)
    st.write("""
    * The stores could be located in densely populated areas hence, distance to nearest competitor has a small influence.
    * Most stores are located around 5km from competitors""")

X = clean_data.loc[:,clean_data.columns != "Sales"]
y = clean_data['Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

st.subheader("Explore the different classifiers to see which is best")

def add_params(classifier_name):
    params = dict()
    if classifier_name == "Random Forest Regressor":
        max_depth = st.sidebar.slider("max_depth",2,10)
        n_estimators = st.sidebar.slider("n_estimators",100,150)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    elif classifier_name == "Decision Tree Regressor":
        max_depth = st.sidebar.slider("max_depth",2,10)
        params["max_depth"] = max_depth
    else:
        max_iter = st.sidebar.slider("max_iter",1000,1500)
        params["max_iter"] = max_iter
    return params

params = add_params(classifier_name)

def get_classifier(classifier_name,params):
    if classifier_name == "Random Forest Regressor":
        model = RandomForestRegressor(max_depth = params["max_depth"],n_estimators = params["n_estimators"],random_state=2)
    elif classifier_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(splitter='random',max_depth=params["max_depth"],random_state=2)
    else:
        model = SGDRegressor(eta0=0.1,fit_intercept=False,shuffle=False,learning_rate='adaptive',random_state=2,max_iter=params["max_iter"])
    return model

def rmse_function(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    return rmse

model = get_classifier(classifier_name,params)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = rmse_function(y_test,y_pred)
st.write(f"The forecasting model running is {classifier_name}")
st.write(f"Accuracy of this model in forecasting store sales is {accuracy}")



