import streamlit as st
import  pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache_data
def lead_taxi_data():
    return  pd.read_csv('.venv/data/texi2019.csv').sample(n=50000, random_state=42)

@st.cache_resource
def lead_model(n_estimators,max_depth):
    if n_estimators=='no limit':
        regr =  RandomForestRegressor(max_depth=max_depth,n_estimators=1000)

    else:
        regr =  RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    return  regr



header= st.container()
dataset=st.container()
modeltraining= st.container()
data_taxi = lead_taxi_data()
pupoction_dist = pd.DataFrame(data_taxi['pickup_location_id'].value_counts())



with header:
    st.title('NYC taxi dataset')
with dataset:
    st.header('Nyc taxi dataset')
    st.text('i  found this datset in ____.com')
    st.write(data_taxi.head())
    st.bar_chart(pupoction_dist)
with modeltraining:
    st.header('Time to train the mode')
    sel_col, sel_disp = st.columns(2)

    max_depth = sel_col.slider(' What should the max depth of the model be?',min_value=10,max_value=100,value=20)
    n_estimators = sel_col.selectbox('How many trees should the be?', options=[100,200,300,'no limit'],index=0)
    sel_col.write(data_taxi.columns)
    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'pickup_location_id')
    regr = lead_model(n_estimators,max_depth)

    X = data_taxi[[input_feature]]
    y= data_taxi['trip_distance']
    regr.fit(X,y)
    prediction = regr.predict(X)
    sel_disp.subheader('Mean squared error of the model is:')
    sel_disp.write(mean_squared_error(y, prediction))
    sel_disp.subheader('R squared score od rhw model is:')
    sel_disp.write(r2_score(y,prediction))


