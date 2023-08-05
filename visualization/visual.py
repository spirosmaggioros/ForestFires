import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, dash_table

def get_layout(fig):
    layout = html.Div(
        children=[
        html.H1(children="Visualization Tool for the project of Forest Fire Danger Detection"),
        dbc.Row(
            dbc.Col(
                    dcc.Graph(id='graph', figure=fig, style={'width': '90vh', 'height': '90vh'}),
                align = "center",
            ),
        ),],
    )

    return layout

