# Import packages
import dash
from dash import Dash, html, callback, Output, Input, State
import pandas as pd
import pickle
import dash_bootstrap_components as dbc
from dash import dcc
import mlflow
import os



# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.MORPH]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Set paths to the model we are using
vehicle_df = pd.read_csv('Cars.csv')
scalar_path = "pages/Model/car-scalar.model"
brand_fuel = "pages/Model/fuel_encoder.model"
brand_enc_path = "pages/Model/car-a2-brand_encoder.model"

#Set mlflow tracking uri
mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
model_name = "st124088_model"
model_version = 1

# loading the models
scaler = pickle.load(open(scalar_path,'rb'))
label_car = pickle.load(open(brand_enc_path, 'rb'))
fuel_car = pickle.load(open(brand_fuel,'rb'))
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")





# get all the possible brand names
brand_cats = list(label_car.categories_[0])
# Getting all fuel values
fuel_cat = list(fuel_car.classes_)
num_cols = ['max_power','year','mileage']
# Map numerical values of 'y' to String representations
y_map = {0: 'Cheap', 1: 'Average', 2: 'Expensive', 3: 'Very expensive'}

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
            dcc.Dropdown(id="brand3",
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
            dcc.Dropdown(id="year3",
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
            dcc.Dropdown(id="fuel3",
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
            dcc.Input(id="mileage3", type="number",
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
            dcc.Input(id="max_power3", type="number",
                      value=0,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
],style={'marginBottom': '20px', 'marginRight': '20px'})


card_predicted_price = dbc.Card([
    dbc.CardBody(
        [
            html.H3("Predicted price is in class: ", className="card-title"),
            html.H3(" ", id="selling_price3", className="card-text", style={"font-weight": "bold;"})
        ]),
],style={'margin-bottom': '20px', 'margin-right': '20px'})



card_reason = dbc.Card([
    dbc.CardBody(
        [
            html.H5(children='Why the A3 Custom Model Excels in Classification', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            html.H3("While the A3 Custom Model may not achieve superior classification metrics, it excels in other important ways:", className="card-text", style={"font-weight": "bold"}),
            html.P("1. **Interpretability:** The custom model offers more interpretable results, aiding in understanding feature impacts in classification tasks.", className="card-text",style={"font-weight": "bold"}),
            html.P("2. **Customization:** It allows for flexible hyperparameter tuning and experimentation specific to classification problems.", className="card-text",style={"font-weight": "bold"}),
            html.P("3. **Regularization:** Supports various regularization techniques to prevent overfitting in classification scenarios.", className="card-text",style={"font-weight": "bold"}),
            html.P("4. **Cross-Validation:** Utilizes k-fold cross-validation for robust performance estimation in classification tasks.", className="card-text",style={"font-weight": "bold"}),
            html.P("5. **Educational Value:** Ideal for learning and research purposes, providing valuable insights into classification concepts and techniques.", className="card-text",style={"font-weight": "bold"}),
            html.P("6. **Transparency:** Offers full control and transparency in model implementation for classification challenges.", className="card-text",style={"font-weight": "bold"}),
            html.P("7. **Feature Engineering:** Encourages domain-specific feature engineering tailored to classification problems.", className="card-text",style={"font-weight": "bold"}),
            html.P("8. **Benchmarking:** Useful for benchmarking and comparing against other classification models and algorithms.", className="card-text",style={"font-weight": "bold"}),
            html.P("9. **Use of One-Hot Encoding:** The brands in this model are one-hot encoded, ensuring ordinality between any brands in classification tasks.", className="card-text",style={"font-weight": "bold"}),
            html.P("10. **Classification of Selling Price:** The model classifies selling price into 4 classes based on equal balanced quartile ranges.", className="card-text",style={"font-weight": "bold"}),

        ]
    ),
],style={'marginBottom': '20px', 'marginRight': '20px'})


layout = dbc.Container([
    dcc.Tab(label="Prediction", children=[
                        dbc.Container([
                            html.Br(),
                            html.H1(children='A3 Assignment', style={'text-align': 'center', 'color':'#531406'}),
                            html.H1(children='Welcome to Chaky Car Company ', style={'text-align': 'center', 'color':'#531406'}),
                            html.H3(children='We predict car-prices based on varierty of features', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H3(children='Please input only those fields that you are familiar with. To ensure accuracy, we fill the null fields with the mean/median/mode values based on our analysis', 
                                    style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
                            html.H2(children='Select the features to predict the car price ', style={'text-align': 'center'}),                
                            html.Hr(),
                            dbc.CardLink([card_manufacturer, card_year, card_power_reading, card_mileage_reading,card_fuel_type]),
                            html.Br(),  
                            html.Div([dbc.Button(id="submit3", children ="Calculate selling Price", style={'text-align': 'center', 'margin-bottom':'20px'})]),                                                  
                            html.Br(),                        
                            dbc.CardLink([card_predicted_price]),
                            html.Br(), 
                            dbc.CardLink(card_reason)
                        ])
                    ])
])
# Define a function to get the input features as X
def get_X(brand, year,max_power, mileage, fuel):
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

    return X.to_numpy(), features
# Define a function to get the predicted selling price based on X
def get_y(X):
    return model.predict(X)

# Setting what we get as input and what we presenting as output in our predict_selling_price function
@callback(
    Output(component_id="selling_price3", component_property="children"),
    Output(component_id="brand3", component_property="value"),
    Output(component_id="year3", component_property="value"),
    Output(component_id="max_power3", component_property="value"),
    Output(component_id="mileage3", component_property="value"),
    Output(component_id="fuel3", component_property="value"),
    #Output(component_id='feature_importance2', component_property='figure'),
    State(component_id="brand3", component_property="value"),
    State(component_id="year3", component_property="value"),
    State(component_id="max_power3", component_property="value"),
    State(component_id="mileage3", component_property="value"),
    State(component_id="fuel3", component_property="value"),
    Input(component_id="submit3", component_property="n_clicks"),
    prevent_initial_call=True
)

# Defining the function along with its input parameters. Note: submit here represents the button that will trigger this function
def predict_selling_price(brand, year,max_power, mileage, fuel, submit):
    #Load X and Y
    X, features = get_X(brand, year,max_power, mileage, fuel)
    predicted_price=get_y(X)[0]
    # Calculating the required prices
    sellling_price = y_map[predicted_price]

    # # Returning our outputs. We are returning the features to fill the form incase of null values
    # We are returning the the class of selling price
    return[f"{sellling_price}"] + list(features.values()) 

# Defining all the columns we are removing
output_col1 = "selling_price"
output_col2 = "km_driven"
output_col3 = "engine"
output_col4 = "seats"
output_col5 = "seller_type"
output_col6 = "owner"
output_col7 = "transmission"