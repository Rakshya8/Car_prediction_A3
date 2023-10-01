from LinearRegression import Normal
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from navbar import create_navbar
from pages import a1
from pages import a2
from pages import a3


# Create the Dash application instance
app = dash.Dash(
    __name__,    
    external_stylesheets=[
        dbc.themes.MORPH,  # Dash Themes CSS
        "https://use.fontawesome.com/releases/v6.2.1/css/all.css",  # Font Awesome Icons CSS
    ],
    title="Machine Learning Asssignment",
    suppress_callback_exceptions=True
)

# Define the layout components
NAVBAR = create_navbar()

# Define the overall app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    NAVBAR,
    html.Div(id='page-content'),
])

# Define callback to update page content based on URL
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/app1':
        return a1.layout
    elif pathname == '/app2':
        return a2.layout
    elif pathname == '/app3':
        return a3.layout
    else:
        return html.H1(children='Welcome to Chaky Car Company. We predict car selling prices for you', style={'text-align': 'center', 'color':'#531406'}),


# Define the server
server = app.server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80',debug=True)
