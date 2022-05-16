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

import pyodbc
from pandas.io import sql

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash.dependencies import Input, Output, State

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

import plotly.graph_objs as go

##############
## Settings ##
##############

fn = 'Combined_output_template.xlsx'

app_title = 'Malaysia ULM Fleet Requirements Forecasting'

imt_cols = ['MYDIN CENTRAL DISTRIBUTION CENTRE - CDC-10T_BOX',
            'MYDIN CENTRAL DISTRIBUTION CENTRE - CDC-1T',
            'MYDIN CENTRAL DISTRIBUTION CENTRE - CDC-20T_BOX',
            'MYDIN CENTRAL DISTRIBUTION CENTRE - CDC-3T',
            'AEON DC-10T_BOX',
            'AEON DC-1T',
            'AEON DC-20T_BOX',
            'AEON DC-3T',
            'WATSON-10T_BOX',
            'WATSON-1T',
            'WATSON-20T_BOX',
            'WATSON-3T',
            'JAYA GROCER DISTRIBUTION CENTER-10T_BOX',
            'JAYA GROCER DISTRIBUTION CENTER-1T',
            'JAYA GROCER DISTRIBUTION CENTER-20T_BOX',
            'JAYA GROCER DISTRIBUTION CENTER-3T',
            'GCH DISTRIBUTION CENTRE- SEPANG-10T_BOX',
            'GCH DISTRIBUTION CENTRE- SEPANG-1T',
            'GCH DISTRIBUTION CENTRE- SEPANG-20T_BOX',
            'GCH DISTRIBUTION CENTRE- SEPANG-3T',
            'TESCO DC(AMBIENT DC)-10T_BOX',
            'TESCO DC(AMBIENT DC)-1T',
            'TESCO DC(AMBIENT DC)-20T_BOX',
            'TESCO DC(AMBIENT DC)-3T',
            'GCH RETAIL (MALAYSIA) SDN BHD-10T_BOX',
            'GCH RETAIL (MALAYSIA) SDN BHD-1T',
            'GCH RETAIL (MALAYSIA) SDN BHD-20T_BOX',
            'GCH RETAIL (MALAYSIA) SDN BHD-3T',
            'AEON BIG (M) SDN. BHD - DC-10T_BOX',
            'AEON BIG (M) SDN. BHD - DC-1T',
            'AEON BIG (M) SDN. BHD - DC-20T_BOX',
            'AEON BIG (M) SDN. BHD - DC-3T']

dt_cols = ['NORTH-10T_BOX',
           'NORTH-20T_BOX',
           'EAST COAST-10T_BOX',
           'EAST COAST-20T_BOX',
           'CENTRAL-10T_BOX',
           'CENTRAL-20T_BOX',
           'SOUTH-10T_BOX',
           'SOUTH-20T_BOX']

######################
## Define Functions ##
######################


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
                     # export_format='xlsx',
                     # export_headers='display',
                     data=df.to_dict('records'))

def render_stackedbar(df, y_list):
    """Render a dataframe as a time series stacked bar chart
    df: The source dataframe; must have a timeindex
    y_list: The list of time series columns to be plotted
    """

    x_axis = df.index

    fig_data = []
    for y in y_list:
        fig_data.append(go.Bar(name=y,
                               x=x_axis,
                               y=df[y],
                               text=df[y],
                               textposition='auto'))

    fig = go.Figure(data=fig_data,
                    layout=go.Layout(height=700))

    fig.update_layout(barmode='stack',
                      legend=dict(yanchor='top',
                                  y=0,
                                  xanchor='left',
                                  x=0))

    return fig


def render_linewitherror(df, y, y_ub, y_lb):
    """Render a dataframe as a time series with error bars
    df: The source dataframe
    y: The column of the point forecast
    y_ub: The column of the upper bound
    y_lb: The column of the lower bound
    """

    x_axis = df.index

    fig = go.Figure(data=go.Scatter(
        x=x_axis,
        y=df[y],
        error_y=dict(
            type='data',
            symmetric=False,
            array=df[y_ub]-df[y],
            arrayminus=df[y]-df[y_lb])
    ))

    return fig


#######################
## Dummy Data Secion ##
#######################

# Load dummy data
data = pd.read_excel(fn)

# Show latest week
data = data.tail(7)

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
    html.Div(id='d-data', style={'display': 'none'}),
    html.Div(id='w-data', style={'display': 'none'}),
    html.Div(id='predictions', style={'display': 'none'}),
    html.H1(children=app_title),
    html.Div(
        children='This tool allows TMS planners to estimate future fleet requirements.'),
    dcc.Tabs(value='dt-tab',
             children=[
                 dcc.Tab(label='IMT',
                         value='imt-tab',
                         children=[
                             html.Div(id='imt-vol-div',
                                      className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='imt-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(data[['Date',
                                                                            'IMT Volume Forecast']],
                                                                      'imt-vol-table',
                                                                      'Date',
                                                                      '%b %d')
                                                   ])
                                      ]),
                             html.Div(id='imt-frf-div',
                                      className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='imt-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='frf-chart',
                                                                 # figure=render_linewitherror(data.set_index('Date'),
                                                                 #                            y='IMT Fleet Forecast (40ft Container)',
                                                                 #                            y_ub='IMT Fleet Forecast (40ft Container) UB',
                                                                 #                            y_lb='IMT Fleet Forecast (40ft Container) LB'))
                                                                 figure=render_stackedbar(data.set_index('Date'),
                                                                                          imt_cols))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='DT',
                         value='dt-tab',
                         children=[
                             html.Div(id='dt-vol-div',
                                      className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                         html.Div(id='dt-vol-container',
                                                  children=[
                                                      # Replace below with data from SQL query
                                                      render_t_table(data[['Date',
                                                                           'DT Volume Forecast']],
                                                                     'dt-vol-table',
                                                                     'Date',
                                                                     '%b %d')
                                                  ])
                                      ]),
                             html.Div(id='dt-frf-div',
                                      className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                         html.Div(id='dt-frf-container',
                                                  children=[
                                                      # Replace below with data from SQL query
                                                      dcc.Graph(id='dt-chart',
                                                                figure=render_stackedbar(data.set_index('Date'),
                                                                                         dt_cols))
                                                  ])
                                      ])
                         ])
             ])
])

###################
## App Callbacks ##
###################


################
## Run Server ##
################

if __name__ == '__main__':
    app.run_server(debug=True, port=8080, host='0.0.0.0')
    # app.run_server(debug=True, port=8080, host='0.0.0.0')