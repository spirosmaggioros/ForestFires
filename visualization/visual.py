import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, dash_table

def get_layout(fig ,fig2,fig3, dropdown_figure):
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
                    dbc.CardImg(src='/Users/spirosmag/Documents/machinevision/ForestFires/visualization/fires.png', top=False, ),
                ],
                    color="info", inverse=True
                ),
                width={"size": 3},
            ),
        ],
            align="left",
            justify="left",
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
                ),],),
        dbc.Row(
            dbc.Col(
                    dcc.Graph(id='graph1', figure=fig, style={'width': '90vh', 'height': '90vh'}),
                    align = "center",
            ), 
        ),
        dbc.Row(
            dbc.Col(
                    dcc.Graph(id='graph2' , figure=fig2 , style={'width' : '90vh' , 'height':'90vh'}),
                    align='center',
                    ),
            ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='graph3' , figure=fig3 , style={'width' : '90vh' , 'height':'90vh'}),
                align='center',
                ),
            ),
        ],
    )

    return layout

