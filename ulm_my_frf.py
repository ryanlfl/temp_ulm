#!/usr/bin/env python
# coding: utf-8

# # Overview

# This program is for Malaysia Unilever Fleet Requiement Forecasting. It will forecast the fleet requirement matrix in coming 7days for segment IMT and DT.

# # Settings

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np

import cx_Oracle


# Import modules or function from sklearn package
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


# Import modules or functions datetime, tqdm, listdir and pathlib
from datetime import datetime, timedelta
from tqdm import tqdm
from os import listdir
import pathlib


# Import pivot_ui for interactive pivot table
from pivottablejs import pivot_ui


# Import Matplotlib library
import matplotlib.pyplot as plt
import matplotlib


# Import modules from ipywidgets library
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import pickle

from tqdm import tqdm


# Import library to connect to MS SQL Server
import pyodbc


# In[2]:


# To increase the number of displayed columns, which enables scrolling data tables horizontally
pd.options.display.max_columns = 99


# In[3]:


# Libraries for prediction
from sklearn.feature_selection import VarianceThreshold,RFECV,SelectKBest,chi2
from datetime import date
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold 
from sklearn import preprocessing, metrics, svm, ensemble, multioutput, model_selection


# In[4]:


# Directory where the data is located
data_path = 'data/'

# Directory where the models are located
model_path = 'models/'


# # Data import and cleansing

# In[5]:


# OTM Login
host_name = 'ORL.LFINFRA.NET'
port_number = '51521'
servicename = 'SHARED.SERVER'

user_name = 'USR_STONESHIBX'
pwd = 'EWPIS3JMC4VCPIFH61U9AKIFIU9CC9'


# In[6]:


#Conncect to OTM database
dsn_tns = cx_Oracle.makedsn(host_name,port_number,service_name = servicename)
conn = cx_Oracle.connect(user=user_name,password=pwd,dsn=dsn_tns)


# In[7]:


#SQL Query for development
sql_query_for_development = """
SELECT
AL1.SHIPMENT_GID,
AL1.SHIPMENT_XID, 
AL1.SERVPROV_GID,
AL1.TOTAL_SHIP_UNIT_COUNT, 
AL1.TOTAL_WEIGHT, 
AL1.TOTAL_VOLUME, 
AL1.FIRST_EQUIPMENT_GROUP_GID, 
AL1.EQUIPMENT_REFERENCE_UNIT_GID,
AL1.USER_DEFINED1_ICON_GID,
AL1.ATTRIBUTE4 AS "ATTRIBUTE4(DRIVER_ID)",
AL1.ATTRIBUTE5 AS "ATTRIBUTE5(VEHICLE_NO)",
AL1.TOTAL_ACTUAL_COST, 
AL1.LOADED_DISTANCE, 
AL1.DEST_LOCATION_GID, 
AL1.ATTRIBUTE1 AS PRINCIPAL,
AL1.ATTRIBUTE2 AS "ATTRIBUTE2(ROUTE)",
AL1.TRANSPORT_MODE_GID,
AL1.TERM_LOCATION_TEXT,
(select location_name from APP_OTMDM_ODS.ODS_LOCATION 
where location_gid = AL1.SOURCE_LOCATION_GID) as "SourceName",
(select location_name from APP_OTMDM_ODS.ODS_LOCATION 
where location_gid = AL1.dest_location_gid) as "DestName",
(select province from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "Region",
(select city from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "City",
(select zone1 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone1",
(select zone2 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone2",
AL2.ATTRIBUTE3 AS "Segment",
AL1.START_TIME,
AL1.END_TIME

FROM APP_OTMDM_MYS_REPORT.ODS_SHIPMENT AL1
INNER JOIN app_otmdm_ods.ods_location AL2 ON AL2.LOCATION_GID =  AL1.dest_LOCATION_GID


WHERE  AL1.ATTRIBUTE1 = 'UNILEVER' 
        AND (AL1.START_TIME BETWEEN add_months (sysdate,-25) and sysdate )
        AND AL1.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL1.PERSPECTIVE='S'
        AND AL2.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL2.ATTRIBUTE3 IN ('DT-LOCAL', 'DT-OUTSTATION', 'IMT', 'IMT-NS')
"""


# In[8]:


#SQL Query for implementation
sql_query = """
SELECT
AL1.SHIPMENT_XID, 
AL1.START_TIME,
AL1.TOTAL_SHIP_UNIT_COUNT, 
AL1.TOTAL_WEIGHT, 
AL1.TOTAL_VOLUME, 
AL1.FIRST_EQUIPMENT_GROUP_GID, 
AL1.ATTRIBUTE1 AS PRINCIPAL,
(select location_name from APP_OTMDM_ODS.ODS_LOCATION 
where location_gid = AL1.dest_location_gid) as "DestName",
(select province from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "Region",
(select city from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "City",
(select zone1 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone1",
(select zone2 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone2",
AL2.ATTRIBUTE3 AS "Segment"


FROM APP_OTMDM_MYS_REPORT.ODS_SHIPMENT AL1
INNER JOIN app_otmdm_ods.ods_location AL2 ON AL2.LOCATION_GID =  AL1.dest_LOCATION_GID


WHERE  AL1.ATTRIBUTE1 = 'UNILEVER' 
        AND (AL1.START_TIME BETWEEN add_months (sysdate,-25) and sysdate )
        AND AL1.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL1.PERSPECTIVE='S'
        AND AL2.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL2.ATTRIBUTE3 IN ('DT-LOCAL', 'DT-OUTSTATION', 'IMT', 'IMT-NS')
"""


# In[9]:


#Run the SQL Query to extract data
raw_shipment_data = pd.read_sql_query(sql_query,conn)
df = raw_shipment_data


# In[10]:


#Disconnect to database
conn.close()


# In[11]:


df.head()


# In[12]:


df.to_csv(data_path+'input_from_SQL_query.csv',index=False)


# In[13]:


#Reformat float number into 2 decimals and reformat start_time format
i = 0
for j in df.dtypes:
    if j == np.float64:
        df.iloc[:,i] = np.round(df.iloc[:,i],decimals=2)
        i+=1
    else:
        i+=1
        continue


df['start_time']=df.START_TIME.apply(lambda x: x.date())
print("min_date:",min(df.start_time))
print("max_date:",max(df.start_time))


# # Functions

# ## Model Performance

# In[14]:


#Measure model performance
def model_performance(result_table,dimension):
    '''
    The result table has date as index. It starts with columns of actual number and end withs columns of forecasted numbers.
    '''
    pr_columns = [dimension,'Date_Time','TruckType','MEAN','SD','CV','MAE','MSE','RMSE','NRMSE_MEAN','NRMSE_SD','NRMSE_IQ','MAPE','SMAPE']
    pr = pd.DataFrame(columns=pr_columns)
    n = int((len(result_table.columns)-1)/2)
    for i in range(len(result_table.columns)-n-1):
        actual = result_table[result_table.columns[i]]
        forecast = result_table[result_table.columns[i+n]]
        current_date_time = datetime.now()
        truck_type = result_table.columns[i]
        a = np.mean(actual)
        s = np.std(actual)
        try:
            cv = s/a
        except ZeroDivisionError:
            cv = np.nan
        mae = mean_absolute_error(actual,forecast)
        mse = metrics.mean_squared_error(actual,forecast)
        rmse = np.sqrt(mse)
        #print(truck_type,mae,mse,rmse,np.std(actual),np.mean(actual))
        nrmse_sd = nrmse(actual,forecast,'sd')
        nrmse_iq = nrmse(actual,forecast,'iq')
        nrmse_mean = nrmse(actual,forecast,'mean')
        mape_value = mape(actual,forecast)
        smape_value =smape(actual,forecast)
        temp_df = pd.DataFrame(data=[[dimension,current_date_time,truck_type,a,s,cv,mae,mse,rmse,nrmse_mean,nrmse_sd,nrmse_iq,mape_value,smape_value]],columns=pr_columns)
        pr = pr.append(temp_df,ignore_index = True)
    return pr


# In[15]:


# mape calculation
def mape(actual,forecast):
    result = []
    for i in range(len(actual)):
        if actual[i] == 0 and forecast[i] != 0:
            result += [1]
        elif actual[i] == 0 and forecast[i] == 0:
            result += [0]
        else:
            result += [np.abs((actual[i] - forecast[i]) / actual[i])]
    #result = np.mean(np.abs((actual - forecast) / actual)) * 100
    return np.mean(result)


# In[16]:


def smape(actual,forecast):
    result = []
    for i in range(len(actual)):
        if actual[i] == 0 and forecast[i] != 0:
            result += [1]
        elif actual[i] == 0 and forecast[i] == 0:
            result += [0]
        else:
            result += [np.abs((actual[i] - forecast[i])) / (np.abs(actual[i])+np.abs(forecast[i]))]
    #result = np.mean(np.abs((actual - forecast) / actual)) * 100
    return np.mean(result)


# In[17]:


def nrmse(actual,forecast,n_type):
    mse = metrics.mean_squared_error(actual,forecast)
    rmse = np.sqrt(mse)
    if n_type == 'sd':
        nrmse = rmse/np.std(actual)
    elif n_type == 'mean':
        nrmse = rmse/np.mean(actual)
    elif n_type == 'maxmin':
        nrmse = rmse/(max(actual)-min(actual))
    elif n_type == 'iq':
        nrmse = rmse / (np.quantile(actual,0.75)-np.quantile(actual,0.25))
    nrmse = np.round(nrmse,decimals=3)
    return nrmse


# ## Data Preparation

# In[18]:


#Prepare ABT table according to specific dimension

def abt(org_df,dimension):
    '''
    Transform the original data set into ABT table by specific dimension provided in 2nd argument.
    By customer, then input string "DestName" in dimension.
    By region, then input string "DestZone2"
    '''
    # Count number of truck by truck type
    cnt_of_trucks = pd.pivot_table(org_df,values=['SHIPMENT_XID'],columns=['FIRST_EQUIPMENT_GROUP_GID'],
                    index = ['start_time',dimension],fill_value=0,
                    aggfunc=lambda x: len(x.unique()))
    # Simplify truck type names
    cnt_col_name = []
    for i in cnt_of_trucks.columns:
        cnt_col_name.append(i[1][i[1].index('.')+1:])
    cnt_of_trucks.columns = cnt_col_name

    cnt_of_trucks.reset_index(level = dimension,inplace=True, col_level=1)
    # Aggregate volume by Customer
    df_vol = pd.pivot_table(org_df,values=['TOTAL_VOLUME'],
                    index = ['start_time',dimension],fill_value=0,aggfunc=np.sum)

    df_vol.reset_index(level = dimension,inplace=True)
    df_vol['Date']=df_vol.index
    df_vol['Year']=df_vol.Date.apply(lambda x: x.year)
    df_vol['Month']=df_vol.Date.apply(lambda x:x.month)
    df_vol['Weekday']=df_vol.Date.apply(lambda x:x.weekday()+1)


    # Concatenate volume and count of trucks
    df_abt = pd.concat([df_vol,cnt_of_trucks],axis=1)
    df_abt.drop(axis=1, labels=['Date'],inplace=True)
    df_abt = df_abt.loc[:,~df_abt.columns.duplicated()]
    return df_abt


# ## Forecasting

# In[19]:


#Multi-output regression
def multioutput_forecast(df_input,x_col,y_col,dimension):
    X=df_input[x_col[1:]]
    y=df_input[y_col]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80)

    model1 = multioutput.MultiOutputRegressor(ensemble.RandomForestRegressor())
    model1.fit(X_train, y_train)

    model2 = multioutput.MultiOutputRegressor(ensemble.AdaBoostRegressor())
    model2.fit(X_train, y_train)

    model3 = multioutput.MultiOutputRegressor(ensemble.ExtraTreesRegressor())
    model3.fit(X_train, y_train)

    model4 = multioutput.MultiOutputRegressor(ensemble.GradientBoostingRegressor())
    model4.fit(X_train, y_train)

    model5 = multioutput.MultiOutputRegressor(ensemble.RandomForestRegressor())
    model5.fit(X_train, y_train)

    # Predict on new data
    y_m1 = model1.predict(X_test)
    y_m2 = model2.predict(X_test)
    y_m3 = model3.predict(X_test)
    y_m4 = model4.predict(X_test)
    y_m5 = model5.predict(X_test)

    #Choose the predictive result per truck type wiht lowest error
    mae_list = [metrics.mean_absolute_error(y_test, y_m1),
                metrics.mean_absolute_error(y_test, y_m2),
                metrics.mean_absolute_error(y_test, y_m3),
                metrics.mean_absolute_error(y_test, y_m4),
                metrics.mean_absolute_error(y_test, y_m5)]
    min_model_num = mae_list.index(min(mae_list))
    
    
    #model list
    model_list = [model1, model2, model3, model4, model5]
    final_model = model_list[min_model_num]
    
    
    #Predicted result list
    result_list = [y_m1, y_m2, y_m3, y_m4, y_m5]
    final_forecast = result_list[min_model_num]
    
    selected_index=X_test.index
    new_col_name = []
    for i in y_col:
        new_col_name.append("pred_"+i)
    pred_result = pd.DataFrame(np.round(final_forecast),index=selected_index,columns=new_col_name)
    result_table = pd.merge(y_test,pred_result,left_index=True,right_index=True)
    result_table = result_table.astype(int)
    result_table = pd.merge(result_table,df_input[[dimension]].loc[selected_index],left_index=True,right_index=True)
    return result_table, final_model


# # Model Training

# ## IMT (by customer)

# In[20]:


#Group fleet data for IMT
IMT = df[df.Segment=='IMT']
IMT_NS = df[df.Segment=='IMT-NS']
df_imt = IMT.append(IMT_NS)


# In[21]:


#Keep top 4 frequently used truck types
index_to_remove = []
for i in list(pd.value_counts(df_imt.FIRST_EQUIPMENT_GROUP_GID).index)[4:]: 
    index_to_remove+=list(df_imt[df_imt.FIRST_EQUIPMENT_GROUP_GID==i].index)
    
df_imt.drop(axis=0,labels=index_to_remove,inplace=True)
print(pd.value_counts(df_imt.FIRST_EQUIPMENT_GROUP_GID))


# In[22]:


# Replaace "WATSON DC" with "WATSON"
temp_index = list(df_imt[df_imt.DestName == 'WATSON DC'].index)
for i in tqdm(temp_index):
    df_imt.DestName.loc[i] = 'WATSON'


# In[23]:


# Aggregate IMT volume by customer
print("From ",min(df_imt.start_time), " to ", max(df_imt.start_time),"\n")
df_imt_vol =  pd.pivot_table(df_imt,values=['TOTAL_VOLUME'],index=['DestName'],aggfunc=np.sum,fill_value=0)
df_imt_vol.sort_values(by=['TOTAL_VOLUME'],ascending=False,inplace=True)
remove_destname_list = list(df_imt_vol.index)[9:] # top 9 DestName
df_imt_vol


#                                 Statistics summary for categorical variables
# "count": the number of records
# "unique": unique count of records
# "top": the value with highest frequency
# "freq": the frequency of the "top" value

# In[24]:


# Remove DestName 'AEON BIG 3RD ...' and 'AEON BIG 2ND DC ...' because these are ad-hoc shipments
index_to_remove = []
print("To remove: ", remove_destname_list)
for i in remove_destname_list: 
    index_to_remove+=list(df_imt[df_imt.DestName==i].index)
df_imt.drop(axis=0,labels=index_to_remove,inplace=True)
print("\nAnalysed DestName: ",pd.DataFrame(set(df_imt.DestName)))


# In[25]:


#IMT ABT (Analytics Base Table)
df_imt_abt = abt(df_imt,"DestName")
df_imt_abt.head()


# In[35]:


dimension = 'DestName'
for i in list(set(df_imt_abt.DestName)):
    print(i)
    df_input = df_imt_abt[df_imt_abt.DestName==i]
    x_col = df_input.columns[:-4]
    y_col = df_input.columns[-4:]
    imt_testing, imt_forecast = multioutput_forecast(df_input,x_col,y_col,dimension)
    pickle_out = open(model_path+"imt_"+i+".pkl","wb")
    pickle.dump(imt_forecast, pickle_out)
    pickle_out.close()
    display(imt_testing.head())


# ## DT (by region)

# In[29]:


#DT fleet forecast by customer
DT_O = df[df.Segment=='DT-OUTSTATION']
DT_L = df[df.Segment=='DT-LOCAL']
df_dt = DT_O.append(DT_L)


# In[30]:


# Remove rows whose truck type is out of top 2 list
index_to_remove = []

for i in list(pd.value_counts(df_dt.FIRST_EQUIPMENT_GROUP_GID).index)[2:]: 
    index_to_remove+=list(df_dt[df_dt.FIRST_EQUIPMENT_GROUP_GID==i].index)
    
df_dt.drop(axis=0,labels=index_to_remove,inplace=True)


# In[ ]:


#DT ABT (Analytics Base Table)
df_dt_abt = abt(df_dt,"DestZone2")
df_dt_abt.head()


# In[36]:


#Forecasting for DT
dimension2 = 'DestZone2'
dt_model_list = {}
for i in list(set(df_dt_abt[dimension2])):
    print(i)
    df_input = df_dt_abt[df_dt_abt[dimension2]==i]
    x_col = df_input.columns[:-2]
    y_col = df_input.columns[-2:]
    dt_testing, dt_forecast = multioutput_forecast(df_input,x_col,y_col,dimension2)
    pickle_out = open(model_path+"dt_"+i+".pkl","wb")
    pickle.dump(dt_forecast, pickle_out)
    pickle_out.close()
    temp_model_name = "dt_forecast_"+i
    dt_model_list[temp_model_name] = dt_forecast
    
    #Forecasting accuracy
    dt_performance = model_performance(dt_testing,dimension2)
    dt_performance.set_index('TruckType',inplace=True)
    dt_performance=np.round(dt_performance,decimals=2)
    display(dt_performance[['MEAN','SD','MAE','MAPE']])
    print("\n")


# # Model Input & Output

# ## IMT

# In[49]:


number_of_day = 7
imt_input = df_imt_abt[df_imt_abt.columns[:-4]].tail(number_of_day*9)


# In[82]:


output_imt = pd.DataFrame(index=imt_input.index)
for i in list(set(imt_input.DestName)):
    temp_df = imt_input[imt_input.DestName==i]
    temp_df = temp_df[temp_df.columns[-4:]]
    temp_output = imt_model_list["imt_forecast_"+i].predict(np.array(temp_df))
    
    temp_col_name = []
    for j in df_imt_abt.columns[-4:]:
        temp_col_name.append(i+"-"+j)
    temp_output = pd.DataFrame(np.round(temp_output),index=temp_df.index,columns=temp_col_name)
    temp_output.astype(np.int64)
    output_imt = output_imt.join(temp_output)
    display(temp_output)


# In[105]:


output_imt.drop_duplicates(inplace=True)


# In[106]:


output_imt


# In[107]:


output_imt.to_excel('IMT_output_template.xlsx')


# In[ ]:





# ## DT

# In[89]:


number_of_day = 7
dt_input = df_dt_abt[df_dt_abt.columns[:-2]].tail(number_of_day*4)
dt_input.head()


# In[97]:


output_dt = pd.DataFrame(index=dt_input.index)
for i in list(set(dt_input.DestZone2)):
    temp_df = dt_input[dt_input.DestZone2==i]
    temp_df = temp_df[temp_df.columns[-4:]]
    temp_output = dt_model_list["dt_forecast_"+i].predict(np.array(temp_df))
    
    temp_col_name = []
    for j in df_dt_abt.columns[-2:]:
        temp_col_name.append(i+"-"+j)
    temp_output = pd.DataFrame(np.round(temp_output),index=temp_df.index,columns=temp_col_name)
    temp_output.astype(np.int64)
    output_dt = output_dt.join(temp_output)
    display(temp_output)


# In[102]:


output_dt.drop_duplicates(inplace=True)


# In[103]:


output_dt


# In[108]:


output_dt.to_excel('DT_output_template.xlsx')


# In[ ]:




