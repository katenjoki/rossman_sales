import time
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
#import plotting and ml functions
from functions import *
#import awesome_streamlit as ast


st.set_page_config(page_title="Rossmann Pharmaceuticals Store Sales", layout="wide")

st.title('Rossman Pharmaceuticals Store Sales Prediction')
st.write("""
         This dashboard has 2 sections that can be seen on the sidebar to the left. \n
         * Plot features - This section contains different feature visualizations.
         * Regression Models (current) - This section runs different regression models and compares their individual evaluation metrics.
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


model_name = st.sidebar.selectbox("ML Models",("Decision Tree Regressor",
                                               "Random Forest Regressor","SGD Regressor"),placeholder="Select a model",index=None)

train_store = pd.read_csv("train_store.csv")
train_store = train_store[:200000]
test_store = pd.read_csv("test_store.csv")


###ML Models
X = train_store.loc[:,train_store.columns != "Sales"]
y = train_store['Sales']
X_train,X_test,y_train,y_test = split_data(train=train_store,test=test_store)
st.header("Machine Learning Models")
st.write("""
         The objective is to predict Sales (target variable) based on the features we have. 
         * The test set didn't have the feature Customer, which had a really strong positive correlation with Sales.
         * As a result, the Customer feature was dropped from the train set, prior to modelling.\n
         On the sidebar, select a model to predict Sales.""")
st.divider()

if model_name:
    st.subheader(model_name)
    if model_name == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=37)
    else:
        model = SGDRegressor(random_state=51)

    start_time = time.time()

    with st.spinner(f'Running the {model_name} model.\nP.S this can take a while as the dataset is large :)'):
        model.fit(X_train,y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time <60:
        st.write(f"Model training time: {elapsed_time:.2f} seconds")
    else:
        st.write(f"Model training time: {elapsed_time/60:.2f} minutes")
    y_pred = model.predict(X_test)
    rmse,mae,r_squared = loss_function(y_test,y_pred)
    r_squared_pct = r_squared * 100
    
    st.text(f"{model_name} Model Evaluation")
    st.write("Evaluation metrics enable us to evaluate model performance. \n Are we able to build a ML model that can accurately predict Sales, given the remaining features?")
    print(type(rmse))
    print('.....................')
    metrics = pd.DataFrame()
    metrics['metric'] = ['RMSE','MAE','Coefficient of Determination (R-squared)']
    metrics['score'] = [rmse,mae,r_squared]
    st.table(metrics)
    st.text("Interpretation")
    st.write(f" - A R squared score of {round(r_squared,2)} meaning that {round(r_squared_pct)}% of the variability in Sales is explained by the features in the {model_name} model.")
    st.write(f" - An RMSE score of {round(rmse)} means that on average the models predictions differ from the actual Sales values by about {round(rmse)} sales.")
   
    st.plotly_chart(plot_feature_importance(X_train,model=model,model_name=model_name))





