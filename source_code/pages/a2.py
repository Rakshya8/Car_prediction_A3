# Import packages
from LinearRegression import Normal
import dash
from dash import Dash, html, callback, Output, Input, State
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.MORPH]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Set paths to the model we are using
vehicle_df = pd.read_csv('Cars.csv')
model_path = 'pages/Model/car-a2-prediction.model'
scalar_path = "pages/Model/a2r-scalar.model"
brand_fuel = "pages/Model/fuel_encoder.model"
brand_enc_path = "pages/Model/car-a2-brand_encoder.model"


# loading the models
model = pickle.load(open(model_path,'rb'))
scaler = pickle.load(open(scalar_path,'rb'))
label_car = pickle.load(open(brand_enc_path, 'rb'))
fuel_car = pickle.load(open(brand_fuel,'rb'))


# get all the possible brand names
brand_cats = list(label_car.categories_[0])
# Getting all fuel values
fuel_cat = list(fuel_car.classes_)
num_cols = ['max_power','year','mileage']


#Default values in case of null or negative input
default_values = {'max_power' : 82.4, 'year' : 2017, 'fuel': 'Diesel', 'brand' : 'Maruti', 'mileage': 19.42}

# Create function for one-hot encoding a feature in dataframe 
def one_hot_transform(encoder, dataframe, feature):

    encoded = encoder.transform(dataframe[[feature]])

    # Transform encoded data arrays into dataframe where columns are based values
    categories = encoder.categories_[0]
    feature_df = pd.DataFrame(encoded.toarray(), columns=categories[1:])
    concat_dataframe = pd.concat([dataframe, feature_df], axis=1)
    
    return concat_dataframe.drop(feature, axis=1)

# App layout
card_manufacturer = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the brand of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Choose Brand'], style={'font-weight': 'bold', "text-align": "center"}),            
            dcc.Dropdown(id="brand2",
                         options=brand_cats,
                         value=brand_cats[0],
                         searchable=True,
                         )
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})

card_year = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the year of manufacture', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Choose Manufacturing Year'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="year2",
                         value = vehicle_df['year'].unique()[0], # get the default value which is the first value in our dataframe
                         options=[{"label": i, "value": i} for i in sorted(vehicle_df['year'].unique())], # Getting all the values of year to show in our dropdown
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "90%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})


card_fuel_type = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please select the fuel type of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Select Fuel Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="fuel2",
                         value=fuel_cat[0],
                         options=fuel_cat,
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "100%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})


card_mileage_reading = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please enter the milaege of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Provide Mileage'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Input(id="mileage2", type="number",
                      value=0,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})


card_power_reading = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Please enter the maximum power of the car', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            dbc.Label(['Provide Maximum Power'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Input(id="max_power2", type="number",
                      value=0,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})


card_predicted_price = dbc.Card([
    dbc.CardBody(
        [
            html.H3("Predicted price in is : ", className="card-title"),
            html.H3(" ", id="selling_price2", className="card-text", style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})

card_upper_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Upper Limit", className="card-title"),
            html.H3("Note: This price is 26% above the Predicted Price.", className="card-title"),
            html.H3(" ", id="upper_range2", className="card-text", style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})


card_lower_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Lower Limit", className="card-title"),
            html.H3("Note: This price is 26% below the Predicted Price.", className="card-title"),
            html.H3(" ", id="lower_range2", className="card-text",style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})

card_reason = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Why the A2 Custom Model Excels', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            html.H3("While the model may not achieve superior MSE or R2 scores, it excels in other important ways:", className="card-text", style={"font-weight": "bold"}),
            html.P("1. **Interpretability:** The custom model offers more interpretable results, aiding in understanding feature impacts.", className="card-text",style={"font-weight": "bold"}),
            html.P("2. **Customization:** It allows for flexible hyperparameter tuning and experimentation.", className="card-text",style={"font-weight": "bold"}),
            html.P("3. **Regularization:** Supports various regularization techniques to prevent overfitting.", className="card-text",style={"font-weight": "bold"}),
            html.P("4. **Cross-Validation:** Utilizes k-fold cross-validation for robust performance estimation.", className="card-text",style={"font-weight": "bold"}),
            html.P("5. **Educational Value:** Ideal for learning and research purposes, providing valuable insights into ML concepts.", className="card-text",style={"font-weight": "bold"}),
            html.P("6. **Transparency:** Offers full control and transparency in model implementation.", className="card-text",style={"font-weight": "bold"}),
            html.P("7. **Feature Engineering:** Encourages domain-specific feature engineering.", className="card-text",style={"font-weight": "bold"}),
            html.P("8. **Benchmarking:** Useful for benchmarking and comparing against other models.", className="card-text",style={"font-weight": "bold"}),
            html.P("8. **Use of one hot encoding:** The brands of this model is one hot encoded compared to the previous model which ensures that there is oradinality between any brands.", className="card-text",style={"font-weight": "bold"}),
        ]
    ),
],style={'marginBottom': '20px', 'marginRight': '20px'})


layout = dbc.Container([
    dcc.Tab(label="Prediction", children=[
                        dbc.Container([
                            html.Br(),
                            html.H1(children='A2 Assignment', style={'text-align': 'center', 'color':'#531406'}),
                            html.H1(children='Welcome to Chaky Car Company ', style={'text-align': 'center', 'color':'#531406'}),
                            html.H3(children='We predict car-prices based on varierty of features', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H3(children='Please input only those fields that you are familiar with. To ensure accuracy, we fill the null fields with the mean/median/mode values based on our analysis', 
                                    style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H2(children='Select the features to predict the car price ', style={'text-align': 'center'}),                
                            html.Hr(),
                            dbc.CardLink([card_manufacturer, card_year, card_power_reading, card_mileage_reading,card_fuel_type]),
                            html.Br(),  
                            html.Div([dbc.Button(id="submit2", children ="Calculate selling Price", style={'text-align': 'center', 'margin-bottom':'20px'})]),                                                  
                            html.Br(),
                            html.H4(children='As the model has around 74% accuracy the upper and lower range of price has also been provided below:', style={'text-align': 'center', 'color':'#531406'}),
                            dbc.CardLink([card_predicted_price]),
                            html.Br(),
                            dbc.CardLink([card_lower_range, card_upper_range]),  
                            dbc.CardLink(card_reason)
                        ])
                    ])
])

# Setting what we get as input and what we presenting as output in our predict_selling_price function
@callback(
    Output(component_id="selling_price2", component_property="children"),
    Output(component_id="upper_range2", component_property="children"),
    Output(component_id="lower_range2", component_property="children"),
    Output(component_id="brand2", component_property="value"),
    Output(component_id="year2", component_property="value"),
    Output(component_id="max_power2", component_property="value"),
    Output(component_id="mileage2", component_property="value"),
    Output(component_id="fuel2", component_property="value"),
    #Output(component_id='feature_importance2', component_property='figure'),
    State(component_id="brand2", component_property="value"),
    State(component_id="year2", component_property="value"),
    State(component_id="max_power2", component_property="value"),
    State(component_id="mileage2", component_property="value"),
    State(component_id="fuel2", component_property="value"),
    Input(component_id="submit2", component_property="n_clicks"),
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
    # Scaling the values
    X[num_cols] = scaler.transform(X[num_cols])
    # Label encoding fuel and brand
    X['fuel'] = fuel_car.transform(X['fuel'])
    X = one_hot_transform(label_car, X, 'brand')

    # Calculating the required prices
    predicted_price=np.round(np.exp(model.predict(X)), 2)
    upper_range = (predicted_price + (.26 * predicted_price))
    lower_range = (predicted_price - (.26 * predicted_price))

    # # Returning our outputs. We are returning the features to fill the form incase of null values
    # # We are also returning the range of selling price as our model is only approx 93%. Hence the actual selling price is in the range of 7% higher and lower than the predicted value
    return[f"{predicted_price[0]}",f"{upper_range[0]:.2f}",f"{lower_range[0]:.2f}"] + list(features.values()) 

# Defining all the columns we are removing
output_col1 = "selling_price"
output_col2 = "km_driven"
output_col3 = "engine"
output_col4 = "seats"
output_col5 = "seller_type"
output_col6 = "owner"
output_col7 = "transmission"
