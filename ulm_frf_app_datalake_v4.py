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
#host_name = 'ORL.LFINFRA.NET'
#port_number = '51521'
#servicename = 'SHARED.SERVER'

#user_name = 'USR_STONESHIBX'
#pwd = 'EWPIS3JMC4VCPIFH61U9AKIFIU9CC9'

# Data lake server name
sv_name = 'wmsBIdm.lfapps.net'
# Data lake database name
db_name = 'Datalake'

# Filename of inputs (change to SQL later)
#fn = 'input_from_SQL_query.csv' # Old format
fn = 'ULM Sample Excel Loader Query Result.csv' # New format from Excel loader

# Filename of SQL query
sql_query = 'excel_loader_frf.sql'

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
        with open(sql_path+sql_query, 'r') as file_otm:
            sql_code = file_otm.read()

        print('Loading Excel Loader forecast data from data lake...')

        # Connect to data lake
        conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                              'Server='+sv_name+';'
                              'Database='+db_name+';'
                              'UID=dsauser;PWD=Analy*9s;')

        try:
            # Query data and save to a dataframe
            df = pd.read_sql_query(sql_code, conn)
            if(df.empty):
                with open(sql_path+'/excel_loader_frf_Xwhere.sql', 'r') as file_otm:
                    sql_code = file_otm.read()
                df = pd.read_sql_query(sql_code, conn)
        except:
            df = pd.DataFrame({})
        finally:    
            # Disconnect from OTM database
            conn.close()

    else:
        # Load CSV
        print('Loading Excel Loader forecast data from CSV...')
        df = pd.read_csv(data_path+fn)

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'],
                                format='%Y/%b/%d')
    # Set Cust_Region to uppercase
    df['Cust_Region'] = df['Cust_Region'].str.upper()

    print('Excel Loader forecast data loaded!')
    return df

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

app = dash.Dash(__name__, url_base_pathname='/ulm-my-frf/',meta_tags=[{"name": "viewport","content": "width=device-width, initial-scale=1"}])
app.title = app_title
server = app.server

from secure_app import SecureApp
sa = SecureApp(app)

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
        #children=['This tool allows TMS planners to estimate future fleet requirements. This uses fleet forecasts uploaded through the ',
        #          dcc.Link(children=['Excel Loader'],
        #                   href='https://ewms.lfapps.net/ExcelLoader/PROD/',
        #                   target='_blank'),
        #          '.']),
        children=['This tool allows TMS planners to estimate future fleet requirements.']),
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
                                        html.P('Kindly click the "Import" button to upload the latest fleed forecasts.'),
                                        dcc.Link(html.Button('Import'), 
                                                 href='https://ewms.lfapps.net/ExcelLoader/PROD/',
                                                 target='_blank'),
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
                                        html.P('Kindly click the "Import" button to upload the latest fleed forecasts.'),
                                        dcc.Link(html.Button('Import'), 
                                                 href='https://ewms.lfapps.net/ExcelLoader/PROD/',
                                                 target='_blank'),
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


sa.requires_authorization_dash(['SCAGROUP'])

###################
## App Callbacks ##
###################

@app.callback(
    [Output('raw-data', 'children'),
     Output('imt-data', 'children'),
     Output('dt-data', 'children')],
    [Input('interval-component', 'n_intervals')]   
    )
def reload_data(n):  
    """Reload data on a regular interval
    n = Count of time intervals
    """ 
    # Load raw data
    df = load_data()

    # Build IMT data from raw data
    df_imt_abt = df.loc[df['Segment']=='IMT']

    # Build DT data from raw data
    df_dt_abt = df.loc[df['Segment']=='DT']

    #print('raw-data')
    #print(df)
    # Save raw data to div for use later
    df = df.to_json(date_format='iso', 
                    orient='split')

    #print('imt-data')
    #print(df_imt_abt)
    # Save IMT data to div for use later
    df_imt_abt = df_imt_abt.to_json(date_format='iso', 
                                    orient='split')

    #print('dt-data')
    #print(df_dt_abt)
    # Save DT data to div for use later
    df_dt_abt = df_dt_abt.to_json(date_format='iso', 
                                  orient='split')

    return df, df_imt_abt, df_dt_abt 

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
    
    # Filter table for relevant columns
    dt_table_df = df_dt_abt[['Date',
                             'Cust_Region',
                             'Volume_Forecast']]
    
    #print('DT Input (Orig):')
    #print(dt_input)

    # Prepare volume data for rendering
    # Pivot by destination
    dt_table_df = dt_table_df.pivot_table(index=['Date'],
                                          columns=['Cust_Region'],
                                          values=['Volume_Forecast'],
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
    [Input('dt-vol-table', 'data'),
     Input('dt-vol-table', 'columns')]
    )
def calc_dt(rows, columns):
    """Calculate DT fleet forecasts
    rows = Rows of data from user-editable volume table
    columns = Column labels from user-editable volume table
    """
    # List of truck types associated with DT
    truck_cols = ['10T_BOX',
                  '20T_BOX']

    # Only proceed if volume table is populated
    #if columns != [{'Date': today, 'DT Volume Forecast': 0}]:
    #print('DT Columns:')     
    #print(columns)
    if columns != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d'), 'id': today.strftime('%b %d')}]:

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
        dt_input = dt_input.rename(columns={'Cust_Region': 'Date'})
        # Convert to date format
        dt_input['Date'] = pd.to_datetime(dt_input['Date'])
        # Melt data
        dt_input = dt_input.melt(id_vars=['Date'],
                                 var_name='Cust_Region',
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
                # Filter by region before loading into model for region
                temp_df = dt_input.loc[dt_input['Cust_Region']==region].copy()
                # Remove region column before loading into model
                temp_df = temp_df.drop(['Cust_Region'], axis=1)
                #print(temp_df)
                temp_output = DT_model[region].predict(np.array(temp_df))

                temp_col_name = []
                for col in truck_cols:
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

        #print('DT FRF Forecast:')
        #print(output_dt)

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

@app.callback(
    Output('imt-vol-container', 'children'),
    [Input('imt-data', 'children')]
    )
def vol_imt(df_imt_abt):
    """Render IMT volume table
    df_imt_abt = JSON file of stored IMT volume data
    """
    # Load data from div
    df_imt_abt = pd.read_json(df_imt_abt, orient='split')
    
    # Filter table for relevant columns
    imt_table_df = df_imt_abt[['Date',
                               'Cust_Region',
                               'Volume_Forecast']]
    
    #print('IMT Input (Orig):')
    #print(imt_input)

    # Prepare volume data for rendering
    # Pivot by destination
    imt_table_df = imt_table_df.pivot_table(index=['Date'],
                                            columns=['Cust_Region'],
                                            values=['Volume_Forecast'],
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
    # Place table of IMT volumes within a div
    imt_vol_div = [imt_vol_table]

    return imt_vol_div

@app.callback(
    Output('imt-frf-container', 'children'),
    [Input('imt-vol-table', 'data'),
     Input('imt-vol-table', 'columns')]
    )
def calc_imt(rows, columns):
    """Calculate IMT fleet forecasts
    rows = Rows of data from user-editable volume table
    columns = Column labels from user-editable volume table
    """
    # List of truck types associated with IMT
    truck_cols = ['10T_BOX',
                  '1T',
                  '20T_BOX',
                  '3T']

    # Only proceed if volume table is populated
    #if columns != [{'Date': today, 'IMT Volume Forecast': 0}]:
    #print('IMT Columns:')     
    #print(columns)
    if columns != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d'), 'id': today.strftime('%b %d')}]:

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
        imt_input = imt_input.rename(columns={'Cust_Region': 'Date'})
        # Convert to date format
        imt_input['Date'] = pd.to_datetime(imt_input['Date'])
        # Melt data
        imt_input = imt_input.melt(id_vars=['Date'],
                                   var_name='Cust_Region',
                                   value_name='TOTAL_VOLUME')
        # Add back date columns
        imt_input['Year'] = imt_input['Date'].apply(lambda x: x.year)
        imt_input['Month'] = imt_input['Date'].apply(lambda x: x.month)
        imt_input['Weekday'] = imt_input['Date'].apply(lambda x: x.weekday()+1)
        # Set date to index
        imt_input = imt_input.set_index('Date')

        print('IMT Input (Loaded):')
        print(imt_input)
        
        # Build dataframe of predictions
        output_imt = pd.DataFrame(index=imt_input.index.unique())
        for region in IMT_list:
            #try:
            # Filter by region before loading into model for region
            temp_df = imt_input.loc[imt_input['Cust_Region']==region].copy()
            # Remove region column before loading into model
            temp_df = temp_df.drop(['Cust_Region'], axis=1)
            print(temp_df)
            temp_output = IMT_model[region].predict(np.array(temp_df))

            temp_col_name = []
            for col in truck_cols:
                temp_col_name.append(region+'-'+col)
            temp_output = pd.DataFrame(np.round(temp_output), 
                                       index=temp_df.index,
                                       columns=temp_col_name)
            temp_output.astype(np.int64)
            print('Temp Output for', region)
            print(temp_output)
            output_imt = output_imt.join(temp_output,
                                       how='left')
            #except:
            #    pass

        # Set NAs in FRF forecast to zero
        output_imt = output_imt.fillna(0)

        print('IMT FRF Forecast:')
        print(output_imt)

        # Render chart of IMT fleet requirements (should be user-editable later)
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

################
## Run Server ##
################

if __name__ == '__main__':
    #app.run_server(debug=False, port=8082, host='0.0.0.0')
    app.run_server(debug=True, port=8082, host='0.0.0.0')
