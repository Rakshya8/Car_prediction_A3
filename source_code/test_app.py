import numpy as np
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import pytest
from pages.a3 import get_X, get_y, predict_selling_price

# Define feature values for testing
feature_vals = ['Maruti', 2017, 82.4, 0, 'Diesel']

# Define labels for expected outputs
labels = ['Cheap', 'Average', 'Expensive', 'Very Expensive']

# Define possible expected outputs
possible_outputs = [label for label in labels]

# Test the get_X and get_y functions
def test_get_Xy():
    # Get feature values and features
    X, features = get_X(*feature_vals)
    
    # Assert the shape and data type of X
    assert X.shape == (1, 35)
    assert X.dtype == np.float64

    # Get predicted y values
    y = get_y(X)

    # Assert the shape of y
    assert y.shape == (1,)

# Test the predict_selling_price callback function
def test_calculate_selling_price_callback():
    # Call the predict_selling_price function with feature values and a dummy n_clicks argument (1)
    output = predict_selling_price(*feature_vals, 1)

    # Assert that the first output is one of the possible outputs (a selling price class)
    assert output[0] in possible_outputs
