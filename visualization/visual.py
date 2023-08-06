import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import plotly.express as px


def set_area_of_interest(meteorological_data , figure):
    if figure == 1:
        fig = px.bar(meteorological_data , x=meteorological_data['Date'].tolist() ,y="danger" , title="Danger level")
    elif figure == 2:
        fig = px.pie(meteorological_data , values='danger' ,names='danger')
    elif figure == 3:
        fig = px.scatter_matrix(meteorological_data , dimensions=['FFMC' , 'DMC' , 'DC' ,'ISI'], color='danger')
    else:
        fig = px.parallel_coordinates(meteorological_data , color='danger' , labels={'FFMC' : 'FFMC' , 'DMC':'DMC' , 'DC':'DC' , 'ISI':'ISI',},
                                      color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

    return fig




def get_layout(fig ,fig2,fig3,fig4, dropdown_figure):
    layout = html.Div(
        children=[
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H2(children="Prediction Model for Forest Fires Danger", style={"textAlign": "center"}, ),
                        html.H5(children="Data Analysis Platform",
                                style={"textAlign": "center", "fontStyle": "italic"}, ),
                    ]),
                    dbc.CardImg(src='fires.png', top=False, ),
                ],
                    color="info", inverse=True
                ),
                width={"size": 3},
            ),
        ],
            align="center",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Markdown(children="Select Case" , style={'font-size': 20}),
                            html.Br(),

                            dcc.Dropdown(dropdown_figure, 'dbscan', id='dropdown-figure',  style={'font-size': 20}),
                            html.Br(),

                        ]),
                    ),
                    width={"size": 3},
                ),
                 dbc.Col(
                    dcc.Graph(id='graph1', figure=fig, style={'width': '90vh', 'height': '90vh'}),
                    align = "center",
                ), 
                ],),
        dbc.Row(
            dbc.Col(
                    dcc.Graph(id='graph2' , figure=fig2 , style={'width' : '90vh' , 'height':'90vh'}),
                    align='center',
                    ),
            ),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='graph3' , figure=fig3 , style={'width' : '100vh' , 'height':'100vh'}),
                align='center',
                ),
             dbc.Col(
                dcc.Graph(id='graph4' , figure=fig4 , style={'width' : '100vh' , 'height':'100vh'}),
                ),
            ],),
    ],)

    return layout

