# Import packages
import dash
from dash import Dash, html, callback, Output, Input, State
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc
import logging
import matplotlib.pyplot as plt

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.MORPH]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Set paths to the model we are using
vehicle_df = pd.read_csv('Cars.csv')
model_path = "pages/Model/car-prediction.model"
scalar_path = "pages/Model/car-scalar.model"
label_path = "pages/Model/car-label.model"
brand_label = "pages/Model/brand-label.model"
brand_fuel = "pages/Model/brand-fuel.model"
rf_Model = "pages/Model/feature_importance.model;"

# loading the models
model = pickle.load(open(model_path,'rb'))
scaler = pickle.load(open(scalar_path,'rb'))
label_car = pickle.load(open(brand_label,'rb'))
fuel_car = pickle.load(open(brand_fuel,'rb'))
rfr = pickle.load(open(rf_Model,'rb'))

# get all brand names
brand_cat = list(label_car.classes_)
fuel_cat = list(fuel_car.classes_)
num_cols = ['max_power','year','mileage']

#Default values in case of null or negative input
default_values = {'max_power' : 82.4, 'year' : 2017, 'fuel': 'Diesel', 'brand' : 'Maruti', 'mileage': 19.42}

# App layout
card_manufacturer = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the brand of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Choose Brand'], style={'font-weight': 'bold', "text-align": "center"}),            
            dcc.Dropdown(id="brand",
                         options=brand_cat,
                         value=brand_cat[0],
                         searchable=True,
                         )
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})

card_year = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the year of manufacture', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Choose Manufacturing Year'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="year",
                         value = vehicle_df['year'].unique()[0],
                         options=[{"label": i, "value": i} for i in sorted(vehicle_df['year'].unique())],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "90%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_fuel_type = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the fuel type of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Select Fuel Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="fuel",
                         value=fuel_cat[0],
                         options=fuel_cat,
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "100%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_mileage_reading = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please enter the milaege of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Provide Mileage'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Input(id="mileage", type="number",
                      value=0,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_power_reading = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please enter the maximum power of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Provide Maximum Power'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Input(id="max_power", type="number",
                      value=0,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_predicted_price = dbc.Card([
    dbc.CardBody(
        [
            html.H3("Predicted price in is : ", className="card-title"),
            html.H3(" ", id="selling_price", className="card-text", style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})

card_upper_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Upper Limit", className="card-title"),
            html.H3("Note: This price is 7% above the Predicted Price.", className="card-title"),
            html.H3(" ", id="upper_range", className="card-text", style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_lower_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Lower Limit", className="card-title"),
            html.H3("Note: This price is 7% below the Predicted Price.", className="card-title"),
            html.H3(" ", id="lower_range", className="card-text",style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})

card_feature_importance = dbc.Card([
    dbc.CardBody(
        [
            dcc.Graph(id='feature_importance', figure={}),
        ]),
])

layout = dbc.Container([
    dcc.Tab(label="Prediction", children=[
                        dbc.Container([
                            html.Br(),
                            html.H1(children='A1 Assignment', style={'text-align': 'center', 'color':'#531406'}),
                            html.H1(children='Welcome to Chaky Car Company ', style={'text-align': 'center', 'color':'#531406'}),
                            html.H3(children='We predict car-prices based on varierty of features', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H3(children='Please input only those fields that you are familiar with. To ensure accuracy, we fill the null fields with the mean/median/mode values based on our analysis', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H2(children='Select the features to predict the car price ', style={'text-align': 'center'}),                
                            html.Hr(),
                            dbc.CardLink([card_manufacturer, card_year, card_power_reading, card_mileage_reading,card_fuel_type]),
                            html.Br(),  
                            html.Div([dbc.Button(id="submit", children ="Calculate selling Price", style={'text-align': 'center', 'margin-bottom':'20px'})]),                                                  
                            html.Br(),
                            html.H4(children='As the model has around 93% accuracy the upper and lower range of price has also been provided below:', style={'text-align': 'center', 'color':'#531406'}),
                            dbc.CardLink([card_predicted_price]),
                            html.Br(),
                            dbc.CardLink([card_lower_range, card_upper_range]),
                            html.Br(),      
                            html.H2(children='The feature importance chart of our model', style={'text-align': 'center'}),                                   
                            dbc.CardLink([card_feature_importance]),
                            html.Br(),

                        ])
                    ])
])

# Setting what we get as input and what we presenting as output in our predict_selling_price function
@callback(
    Output(component_id="selling_price", component_property="children"),
    Output(component_id="upper_range", component_property="children"),
    Output(component_id="lower_range", component_property="children"),
    Output(component_id="brand", component_property="value"),
    Output(component_id="year", component_property="value"),
    Output(component_id="max_power", component_property="value"),
    Output(component_id="mileage", component_property="value"),
    Output(component_id="fuel", component_property="value"),
    Output(component_id='feature_importance', component_property='figure'),
    State(component_id="brand", component_property="value"),
    State(component_id="year", component_property="value"),
    State(component_id="max_power", component_property="value"),
    State(component_id="mileage", component_property="value"),
    State(component_id="fuel", component_property="value"),
    Input(component_id="submit", component_property="n_clicks"),
    prevent_initial_call=True
)

# Defining the function along with its input parameters. Note: submit here represents the button that will trigger this function
def predict_selling_price(brand, year,max_power, mileage, fuel, submit):
    features = {
        'brand': brand,
        'year': year,
        'max_power': max_power,
        'mileage':mileage,
        'fuel': fuel,
    }

    # For verification of null and invalid inputs such as negatives

    for feature in features:
        # Checking for null
        if not features[feature]:
            features[feature]=default_values[feature]
            # Checking for negative values
        elif feature in num_cols:
            if features[feature]<0:
                features[feature]= default_values[feature]
    X= pd.DataFrame(features, index=[0])
    # Scaling the values
    X[num_cols]= scaler.transform(X[num_cols])
    # Label encoding fuel and brand
    X['fuel']= fuel_car.transform(X['fuel'])
    X['brand']= label_car.transform(X['brand'])
    # Calculating the required prices
    predicted_price=np.round(np.exp(model.predict(X)),2)
    upper_range = (predicted_price + (.07 * predicted_price))
    lower_range = (predicted_price - (.07 * predicted_price))
    importances = rfr.feature_importances_
    features_a = x_features.columns
    x_values = list(features_a)
    fig = px.bar(x=x_values, y=importances, title='Random Forest Variables Importance',
                 labels={'x': 'Features', 'y': 'Feature Weightage'})
    fig.update_xaxes(tickangle=90, tickmode='array', tickvals=features_a)
    # # Returning our outputs. We are returning the features to fill the form incase of null values
    # # We are also returning the range of selling price as our model is only approx 93%. Hence the actual selling price is in the range of 7% higher and lower than the predicted value
    return[f"{predicted_price[0]}",f"{upper_range[0]:.2f}",f"{lower_range[0]:.2f}"] + list(features.values())+[fig]


# Working for the feature importance chart
# Calling the final encoded dataframe
after_label_encoding = pd.read_csv('vehicle_final_le.csv')
# Defining all the columns we are removing
output_col1 = "selling_price"
output_col2 = "km_driven"
output_col3 = "engine"
output_col4 = "seats"
output_col5 = "seller_type"
output_col6 = "owner"
output_col7 = "transmission"
feature_cols = after_label_encoding.columns.tolist()

# Removing them
feature_cols.remove(output_col1)
feature_cols.remove(output_col2)
feature_cols.remove(output_col3)
feature_cols.remove(output_col4)
feature_cols.remove(output_col5)
feature_cols.remove(output_col6)
feature_cols.remove(output_col7)

x_features = after_label_encoding[feature_cols]

# Callback to check Feature Importance
# @app.callback(
#     Output(component_id='feature_importance', component_property='figure'),
#     Input(component_id="submit", component_property="n_clicks"),
# )

# # Function to create the importance chart
# def feature_importance(submit):
#     importances = rfr.feature_importances_
#     features = x_features.columns
#     x_values = list(features)
#     fig = px.bar(x=x_values, y=importances, title='Random Forest Variables Importance',
#                  labels={'x': 'Features', 'y': 'Feature Weightage'})
#     fig.update_xaxes(tickangle=90, tickmode='array', tickvals=features)

#     return fig

# Initilization of the web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000',debug=True)