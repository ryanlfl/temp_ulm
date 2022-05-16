######################
## Import Libraries ##
######################

import base64
import io
import math

import urllib
import json

import numpy as np
import pandas as pd

from datetime import timedelta, datetime

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash.dependencies import Input, Output, State

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

import plotly.graph_objs as go

import pyodbc
from pandas.io import sql
import cx_Oracle


##############
## Settings ##
##############

# Memoize date today
today = pd.to_datetime(datetime.today())

# Use SQL?
use_SQL = True

# OTM Login
host_name = 'ORL.LFINFRA.NET'
port_number = '51521'
servicename = 'SHARED.SERVER'

user_name = 'USR_STONESHIBX'
pwd = 'EWPIS3JMC4VCPIFH61U9AKIFIU9CC9'

# Data lake server name
sv_name = 'wmsBIdm.lfapps.net'
# Data lake database name
db_name = 'Datalake'

# Filename of inputs (change to SQL later)
fn = 'input_from_SQL_query.csv'

# Filename of SQL query
sql_query = 'otm_ulm_frf_datalake.sql'

# App title
app_title = 'Malaysia ULM Fleet Requirements Forecasting'

# Number of days to forecast
number_of_day = 7

# Directory where the data is located
data_path = 'data/'

# Directory where the models are located
model_path = 'models/'

# Directory where the SQL query is located
sql_path = 'SQL/'

# List of IMT regions
IMT_list = ['AEON BIG (M) SDN. BHD - DC',
            'AEON DC',
            'GCH DISTRIBUTION CENTRE- SEPANG',
            'GCH RETAIL (MALAYSIA) SDN BHD',
            'MYDIN CENTRAL DISTRIBUTION CENTRE - CDC',
            'TESCO GROCERY (AMBIENT DC)',
            'WATSON',
            'TESCO DC(AMBIENT DC)',
            'JAYA GROCER DISTRIBUTION CENTER']

# List of DT regions
DT_list = ['SOUTH', 
           'NORTH', 
           'CENTRAL',
           'EAST COAST']

######################
## Define functions ##
######################

# Functions for back-end calculations

def load_data():
    """Load data from CSV. Will change to SQL later.
    """ 
    if use_SQL:
        # Read OTM SQL query 
        file_otm = open(sql_path+sql_query, 'r')
        sql_code = file_otm.read()
        file_otm.close()

        print('Loading historical data from data lake...')

        # Connect to data lake
        conn = pyodbc.connect('Driver={SQL Server};'
                              'Server='+sv_name+';'
                              'Database='+db_name+';'
                              'Trusted_Connection=yes;')

        # Query data and save to a dataframe
        df = pd.read_sql_query(sql_code, conn)
        
        # Disconnect from OTM database
        conn.close()

    else:
        # Load CSV
        print('Loading historical data from CSV...')
        df = pd.read_csv(data_path+fn)

    # Reformat float number into 2 decimals and reformat start_time format
    i = 0
    for j in df.dtypes:
        if j == np.float64:
            df.iloc[:,i] = np.round(df.iloc[:,i],decimals=2)
            i += 1
        else:
            i += 1
            continue
    df['start_time'] = pd.to_datetime(df['START_TIME']).dt.date
    print('Historical data loaded!')
    return df

def abt(org_df, dimension):
    """Transform the original data set into ABT by specific dimension provided in 2nd argument. By customer, then input string "DestName" in dimension. By region, then input string "DestZone2"
    org_df = The original dataframe
    dimension = The dimension to pivot on
    """
    # Count number of truck by truck type
    cnt_of_trucks = pd.pivot_table(org_df, 
                                   values=['SHIPMENT_XID'],
                                   columns=['FIRST_EQUIPMENT_GROUP_GID'],
                                   index = ['start_time', 
                                            dimension], 
                                   fill_value=0,
                                   aggfunc=lambda x: len(x.unique()))
    # Simplify truck type names
    cnt_col_name = []
    for i in cnt_of_trucks.columns:
        cnt_col_name.append(i[1][i[1].index('.')+1:])
    cnt_of_trucks.columns = cnt_col_name

    cnt_of_trucks.reset_index(level=dimension, inplace=True, col_level=1)
    # Aggregate volume by Customer
    df_vol = pd.pivot_table(org_df,
                            values=['TOTAL_VOLUME'],
                            index=['start_time', dimension],
                            fill_value=0,
                            aggfunc=np.sum)

    df_vol.reset_index(level=dimension, inplace=True)
    df_vol['Date'] = df_vol.index

    # Concatenate volume and count of trucks
    df_abt = pd.concat([df_vol,cnt_of_trucks], axis=1)
    df_abt.drop(axis=1, labels=['Date'], inplace=True)
    df_abt = df_abt.loc[:, ~df_abt.columns.duplicated()]
    return df_abt

def prep_imt(df):
    """Prepare data to be fed into the IMT model
    df = The original data from SQL (or a CSV extract)
    """
    # Group fleet data for IMT
    print('Preparing IMT input data...')
    IMT = df[df.Segment=='IMT']
    IMT_NS = df[df.Segment=='IMT-NS']
    df_imt = IMT.append(IMT_NS)

    # Keep top 4 frequently used truck types
    index_to_remove = []
    for i in list(pd.value_counts(df_imt.FIRST_EQUIPMENT_GROUP_GID).index)[4:]:
        index_to_remove += list(df_imt[df_imt.FIRST_EQUIPMENT_GROUP_GID == i].index)

    df_imt.drop(axis=0, labels=index_to_remove, inplace=True)

    # Replace "WATSON DC" with "WATSON"
    temp_index = list(df_imt[df_imt.DestName == 'WATSON DC'].index)
    for i in temp_index:
        df_imt.DestName.loc[i] = 'WATSON'

    # Aggregate IMT volume by customer
    df_imt_vol = pd.pivot_table(df_imt, values=['TOTAL_VOLUME'], index=['DestName'], aggfunc=np.sum, fill_value=0)
    df_imt_vol.sort_values(by=['TOTAL_VOLUME'], ascending=False, inplace=True)
    remove_destname_list = list(df_imt_vol.index)[9:]  # Top 9 DestName

    # Remove DestName 'AEON BIG 3RD ...' and 'AEON BIG 2ND DC ...' because these are ad-hoc shipments
    index_to_remove = []
    for i in remove_destname_list:
        index_to_remove += list(df_imt[df_imt.DestName == i].index)
    df_imt.drop(axis=0, labels=index_to_remove, inplace=True)

    print('IMT input data ready!')
    # IMT ABT (Analytics Base Table)
    return abt(df_imt, "DestName")

def prep_dt(df):
    """Prepare data to be fed into the DT model
    df = The original data from SQL (or a CSV extract)
    """
    # DT fleet forecast by customer
    print('Preparing DT input data')
    DT_O = df[df.Segment=='DT-OUTSTATION']
    DT_L = df[df.Segment=='DT-LOCAL']
    df_dt = DT_O.append(DT_L)

    # Remove rows whose truck type is out of top 2 list
    index_to_remove = []

    for i in list(pd.value_counts(df_dt.FIRST_EQUIPMENT_GROUP_GID).index)[2:]: 
        index_to_remove+=list(df_dt[df_dt.FIRST_EQUIPMENT_GROUP_GID==i].index)
        
    df_dt.drop(axis=0,labels=index_to_remove,inplace=True)

    print('DT input data ready!')
    # DT ABT (Analytics Base Table)
    return abt(df_dt, "DestZone2")

# Functions for front-end rendering

def render_table(df, id):
    """Render a dataframe as an HTML table in Dash
    df: The source dataframe
    id: The element ID
    """
    return DataTable(id=id,
                     columns=[{'name': i, 'id': i} for i in df.columns],
                     # export_format='xlsx',
                     # export_headers='display',
                     data=df.to_dict('records'))

def render_t_table(df, id, colname_column, dateindex_format=False):
    """Render a dataframe as a transposed HTML table in Dash
    df: The source dataframe
    id: The element ID,
    colname_column: The column with entries to be used as column names
    dateindex_format: The format to be used for data columns
    """
    if dateindex_format:
        df[colname_column] = df[colname_column].dt.strftime(dateindex_format)
    df[colname_column] = df[colname_column].apply(str)
    df = df.set_index(colname_column)
    df = df.T
    df = df.reset_index()
    return DataTable(id=id,
                     columns=[{'name': i, 'id': i} for i in df.columns],
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_editable_t_table(df, id, colname_column, dateindex_format=False):
    """Render a dataframe as a transposed HTML table in Dash
    df: The source dataframe
    id: The element ID,
    colname_column: The column with entries to be used as column names
    dateindex_format: The format to be used for data columns
    """
    if dateindex_format:
        df[colname_column] = df[colname_column].dt.strftime(dateindex_format)
    df[colname_column] = df[colname_column].apply(str)
    df = df.set_index(colname_column)
    df = df.T
    df = df.reset_index()
    return DataTable(id=id,
                     editable=True,
                     style_data_conditional=[{
                        'if': {'column_editable': True},
                        'backgroundColor': 'rgb(100, 100, 100)',
                        'color': 'white'
                     }],
                     columns=[{'name': i, 'id': i, 'editable': (i in df.columns)} for i in df.columns],  
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_stackedbar(df, y_list=None):
    """Render a dataframe as a time series stacked bar chart
    df: The source dataframe; must have a timeindex
    y_list: The list of time series columns to be plotted
    """

    x_axis = df.index

    if y_list is None:
        y_list = df.columns

    fig_data = []
    for y in y_list:
        fig_data.append(go.Bar(name=y,
                               x=x_axis,
                               y=df[y],
                               text=df[y],
                               textposition='auto'))

    fig = go.Figure(data=fig_data,
                    layout=go.Layout(height=700))

    fig.update_xaxes(dtick="D1", # Show labels for all dates
                     tickformat="%b %d\n%Y")

    fig.update_layout(barmode='stack',
                      margin=dict(l=0, r=0, t=0, b=0)), # Decrease chart padding
                      #legend=dict(yanchor='top',
                      #            y=-1,
                      #            xanchor='left',
                      #            x=0))

    return fig

####################
## Load constants ##
####################

# Load pre-trained IMT models
IMT_model = {}
for dest in IMT_list:
    IMT_model[dest] = pickle.load(open('models/imt_'+dest+'.pkl', 'rb'))

# Load pre-trained IMT models
DT_model = {}
for dest in DT_list:
    DT_model[dest] = pickle.load(open('models/dt_'+dest+'.pkl', 'rb'))

################
## App Layout ##
################

external_stylesheets = ['assets/style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = app_title

app.layout = html.Div(children=[
    dcc.Interval(id='interval-component',
                 interval=24*60*60*1000,  # 1 day in milliseconds
                 n_intervals=0),
    html.Div(id='raw-data', style={'display': 'none'}),
    html.Div(id='imt-data', style={'display': 'none'}),
    html.Div(id='dt-data', style={'display': 'none'}),
    html.Div(id='imt-data-2', style={'display': 'none'}),
    html.Div(id='dt-data-2', style={'display': 'none'}),
    #html.Div(id='imt-predictions', style={'display': 'none'}),
    #html.Div(id='dt-predictions', style={'display': 'none'}),
    html.H1(children=app_title),
    html.Div(
        children=['This tool allows TMS planners to estimate future fleet requirements. This uses fleet forecasts uploaded through the ',
                  dcc.Link(children=['Excel Loader'],
                           href='https://ewms.lfapps.net/ExcelLoader/PROD/',
                           target='_blank'),
                  '.']),
    dcc.Tabs(value='dt-tab',
             children=[
                 dcc.Tab(label='IMT',
                         value='imt-tab',
                         children=[
                             html.Div(id='imt-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='imt-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'IMT Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'imt-vol-table',
                                                                      'Date',
                                                                      '%b %d')
                                                   ])
                                      ]),
                             html.Div(id='imt-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='imt-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='imt-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'IMT Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='DT',
                         value='dt-tab',
                         children=[
                             html.Div(id='dt-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                         html.Div(id='dt-vol-container',
                                                  children=[
                                                      # Replace below with data from SQL query
                                                      render_t_table(pd.DataFrame({'Date': today, 
                                                                                   'DT Volume Forecast': 0},
                                                                                   index=[0]),
                                                                     'dt-vol-table',
                                                                     'Date',
                                                                     '%b %d')
                                                  ])
                                      ]),
                             html.Div(id='dt-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                         html.Div(id='dt-frf-container',
                                                  children=[
                                                      # Replace below with data from SQL query
                                                      dcc.Graph(id='dt-chart',
                                                                figure=render_stackedbar(pd.DataFrame({'DT Fleet Forecast': 0},
                                                                                                      index=[today])))
                                                  ])
                                      ])
                         ])
             ])
])

###################
## App Callbacks ##
###################
@app.callback(
    [Output('raw-data', 'children'),
     Output('imt-data', 'children'),
     Output('dt-data', 'children'),
     Output('imt-data-2', 'children'),
     Output('dt-data-2', 'children')],
    [Input('interval-component', 'n_intervals')]   
    )
def reload_data(n):  
    """Reload data on a regular interval
    n = Count of time intervals
    """ 
    # Load raw data
    df = load_data()

    # Build IMT data from raw data
    df_imt_abt = prep_imt(df)

    # Build DT data from raw data
    df_dt_abt = prep_dt(df)

    # Save raw data to div for use later
    df = df.to_json(date_format='iso', 
                    orient='split')

    # Save IMT data to div for use later
    df_imt_abt = df_imt_abt.to_json(date_format='iso', 
                                    orient='split')

    # Save DT data to div for use later
    df_dt_abt = df_dt_abt.to_json(date_format='iso', 
                                  orient='split')

    return df, df_imt_abt, df_dt_abt, df_imt_abt, df_dt_abt

@app.callback(
    Output('imt-vol-container', 'children'),
    [Input('imt-data', 'children')]
    )
def vol_imt(df_imt_abt):
    """Render DT volume table
    df_dt_abt = JSON file of stored DT volume data
    """
    # Load data from div
    df_imt_abt = pd.read_json(df_imt_abt, orient='split')
    
    # Use latest few days as input for forecast 
    imt_input = df_imt_abt[df_imt_abt.columns[:-2]].tail(number_of_day*4)
    
    #print('DT Input (Orig):')
    #print(imt_input)

    # Prepare volume data for rendering
    imt_table_df = imt_input.reset_index()
    imt_table_df = imt_table_df.rename(columns={'index': 'Date'})
    # Pivot by destination
    imt_table_df = imt_table_df.pivot_table(index=['Date'],
                                            columns=['DestName'],
                                            values=['TOTAL_VOLUME'],
                                            aggfunc=sum)
    # Flatten hierarchical index
    imt_table_df.columns = imt_table_df.columns.get_level_values(1)
    # Reset index to display date
    imt_table_df = imt_table_df.reset_index()
    # Round off volume table
    imt_table_df = np.round(imt_table_df, 2)
    # Set NAs in volume table to zero
    imt_table_df = imt_table_df.fillna(0)
    #print('IMT Volume Table (Saved):')
    #print(imt_table_df)
    # Render table of IMT volumes (should be user-editable later)
    imt_vol_table = render_editable_t_table(imt_table_df,
                                            'imt-vol-table',
                                            'Date',
                                            dateindex_format='%b %d %Y')
    # Place table of DT volumes within a div
    imt_vol_div = [imt_vol_table]

    return imt_vol_div

@app.callback(
    Output('imt-frf-container', 'children'),
    [Input('imt-data-2', 'children'),
     Input('imt-vol-table', 'data'),
     Input('imt-vol-table', 'columns')]
    )
def calc_imt(df_imt_abt, rows, columns):
    """Calculate DT fleet forecasts
    df_imt_abt = JSON file of stored IMT volume data
    rows = Rows of data from user-editable volume table
    columns = Column labels from user-editable volume table
    """

    # Only proceed if volume table is populated
    #if columns != [{'Date': today, 'IMT Volume Forecast': 0}]:     
    if columns != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d'), 'id': today.strftime('%b %d')}]:      
        # Load data from div
        df_imt_abt = pd.read_json(df_imt_abt, orient='split')  

        # Load data from user-editable table
        imt_input = pd.DataFrame(rows, columns=[c['name'] for c in columns])

        # Transpose data
        imt_input = imt_input.T
        # Reset index
        imt_input = imt_input.reset_index()
        # Get columns from first row
        imt_input.columns = list(imt_input.loc[0])
        # Drop first row
        imt_input = imt_input[1:]
        # Rename date column
        imt_input = imt_input.rename(columns={'DestName': 'Date'})
        # Convert to date format
        imt_input['Date'] = pd.to_datetime(imt_input['Date'])
        # Melt data
        imt_input = imt_input.melt(id_vars=['Date'],
                                   var_name='DestName',
                                   value_name='TOTAL_VOLUME')
        # Add back date columns
        imt_input['Year'] = imt_input['Date'].apply(lambda x: x.year)
        imt_input['Month'] = imt_input['Date'].apply(lambda x: x.month)
        imt_input['Weekday'] = imt_input['Date'].apply(lambda x: x.weekday()+1)
        # Set date to index
        imt_input = imt_input.set_index('Date')

        #print('IMT Input (Loaded):')
        #print(imt_input)
        
        # Build dataframe of predictions
        output_imt = pd.DataFrame(index=imt_input.index.unique())
        for dest in IMT_list:
            try:
                temp_df = imt_input[imt_input['DestName']==dest]
                temp_df = temp_df[temp_df.columns[-4:]]
                temp_output = IMT_model[dest].predict(np.array(temp_df))

                temp_col_name = []
                for col in df_imt_abt.columns[-4:]:
                    temp_col_name.append(dest+'-'+col)
                temp_output = pd.DataFrame(np.round(temp_output), 
                                           index=temp_df.index,
                                           columns=temp_col_name)
                temp_output.astype(np.int64)
                #print('Temp Output for', dest)
                #print(temp_output)
                output_imt = output_imt.join(temp_output,
                                             how='left')
            except:
                pass

        # Set NAs in FRF forecast to zero
        output_imt = output_imt.fillna(0)

        print('IMT FRF Forecast:')
        print(output_imt)

        # Render chart of IMT fleet requirements
        imt_frf_chart = dcc.Graph(id='imt-chart',
                                  figure=render_stackedbar(output_imt))

        # Place chart of IMT fleet requiremtns within a div
        imt_frf_div = [imt_frf_chart]

        # Render table of IMT fleet requirements
        imt_frf_table = render_t_table(output_imt.reset_index(),
                                       'imt-frf-table',
                                       'Date',
                                       dateindex_format='%b %d %Y')

        # Add table to final div
        imt_frf_div.append(imt_frf_table)

    else:
        # Default to blank graph if volume data is still loading
        imt_frf_div = [dcc.Graph(id='imt-chart',
                                 figure=render_stackedbar(pd.DataFrame({'IMT Fleet Forecast': 0},
                                                                       index=[today])))]

    return imt_frf_div

@app.callback(
    Output('dt-vol-container', 'children'),
    [Input('dt-data', 'children')]
    )
def vol_dt(df_dt_abt):
    """Render DT volume table
    df_dt_abt = JSON file of stored DT volume data
    """
    # Load data from div
    df_dt_abt = pd.read_json(df_dt_abt, orient='split')
    
    # Use latest few days as input for forecast 
    dt_input = df_dt_abt[df_dt_abt.columns[:-2]].tail(number_of_day*4)
    
    #print('DT Input (Orig):')
    #print(dt_input)

    # Prepare volume data for rendering
    dt_table_df = dt_input.reset_index()
    dt_table_df = dt_table_df.rename(columns={'index': 'Date'})
    # Pivot by destination
    dt_table_df = dt_table_df.pivot_table(index=['Date'],
                                          columns=['DestZone2'],
                                          values=['TOTAL_VOLUME'],
                                          aggfunc=sum)
    # Flatten hierarchical index
    dt_table_df.columns = dt_table_df.columns.get_level_values(1)
    # Reset index to display date
    dt_table_df = dt_table_df.reset_index()
    # Round off volume table
    dt_table_df = np.round(dt_table_df, 2)
    # Set NAs in volume table to zero
    dt_table_df = dt_table_df.fillna(0)
    #print('DT Volume Table (Saved):')
    #print(dt_table_df)
    # Render table of DT volumes (should be user-editable later)
    dt_vol_table = render_editable_t_table(dt_table_df,
                                           'dt-vol-table',
                                           'Date',
                                           dateindex_format='%b %d %Y')
    # Place table of DT volumes within a div
    dt_vol_div = [dt_vol_table]

    return dt_vol_div

@app.callback(
    Output('dt-frf-container', 'children'),
    [Input('dt-data-2', 'children'),
     Input('dt-vol-table', 'data'),
     Input('dt-vol-table', 'columns')]
    )
def calc_dt(df_dt_abt, rows, columns):
    """Calculate DT fleet forecasts
    df_dt_abt = JSON file of stored DT volume data
    rows = Rows of data from user-editable volume table
    columns = Column labels from user-editable volume table
    """

    # Only proceed if volume table is populated
    #if columns != [{'Date': today, 'DT Volume Forecast': 0}]:
    #print('DT Columns:')     
    #print(columns)
    if columns != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d'), 'id': today.strftime('%b %d')}]:   
        # Load data from div
        df_dt_abt = pd.read_json(df_dt_abt, orient='split')  

        # Load data from user-editable table
        dt_input = pd.DataFrame(rows, columns=[c['name'] for c in columns])

        # Transpose data
        dt_input = dt_input.T
        # Reset index
        dt_input = dt_input.reset_index()
        # Get columns from first row
        dt_input.columns = list(dt_input.loc[0])
        # Drop first row
        dt_input = dt_input[1:]
        # Rename date column
        dt_input = dt_input.rename(columns={'DestZone2': 'Date'})
        # Convert to date format
        dt_input['Date'] = pd.to_datetime(dt_input['Date'])
        # Melt data
        dt_input = dt_input.melt(id_vars=['Date'],
                                 var_name='DestZone2',
                                 value_name='TOTAL_VOLUME')
        # Add back date columns
        dt_input['Year'] = dt_input['Date'].apply(lambda x: x.year)
        dt_input['Month'] = dt_input['Date'].apply(lambda x: x.month)
        dt_input['Weekday'] = dt_input['Date'].apply(lambda x: x.weekday()+1)
        # Set date to index
        dt_input = dt_input.set_index('Date')

        #print('DT Input (Loaded):')
        #print(dt_input)
        
        # Build dataframe of predictions
        output_dt = pd.DataFrame(index=dt_input.index.unique())
        for region in DT_list:
            try:
                temp_df = dt_input[dt_input['DestZone2']==region]
                temp_df = temp_df[temp_df.columns[-4:]]
                temp_output = DT_model[region].predict(np.array(temp_df))

                temp_col_name = []
                for col in df_dt_abt.columns[-2:]:
                    temp_col_name.append(region+'-'+col)
                temp_output = pd.DataFrame(np.round(temp_output), 
                                           index=temp_df.index,
                                           columns=temp_col_name)
                temp_output.astype(np.int64)
                #print('Temp Output for', region)
                #print(temp_output)
                output_dt = output_dt.join(temp_output,
                                           how='left')
            except:
                pass

        # Set NAs in FRF forecast to zero
        output_dt = output_dt.fillna(0)

        print('DT FRF Forecast:')
        print(output_dt)

        # Render chart of DT fleet requirements (should be user-editable later)
        dt_frf_chart = dcc.Graph(id='dt-chart',
                                 figure=render_stackedbar(output_dt))
        # Place chart of DT fleet requiremtns within a div
        dt_frf_div = [dt_frf_chart]

        # Render table of DT fleet requirements
        dt_frf_table = render_t_table(output_dt.reset_index(),
                                      'dt-frf-table',
                                      'Date',
                                      dateindex_format='%b %d %Y')

        # Add table to final div
        dt_frf_div.append(dt_frf_table)

    else:
        # Default to blank graph if volume data is still loading
        dt_frf_div = [dcc.Graph(id='dt-chart',
                                figure=render_stackedbar(pd.DataFrame({'DT Fleet Forecast': 0},
                                                                      index=[today])))]

    return dt_frf_div


################
## Run Server ##
################

if __name__ == '__main__':
    #app.run_server(debug=False, port=8082, host='0.0.0.0')
    app.run_server(debug=True, port=8082, host='0.0.0.0')