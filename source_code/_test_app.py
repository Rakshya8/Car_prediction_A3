import numpy as np
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import pytest
from pages/a3 import get_X, get_y, calculate_selling_price

feature_vals = [82.4, 2015, 'Diesel', 'Maruti']
labels = ['Cheap', 'Average', 'Expensive', 'Very Expensive']
possible_outputs = [f"Selling price is: {label}" for label in labels]

def test_get_Xy():
    X, features = get_X(*feature_vals)
    assert X.shape == (1, 34)
    assert X.dtype == np.float64

    y = get_y(X)
    assert y.shape == (1,)

def test_calculate_selling_price_callback():
    output = calculate_selling_price(*feature_vals, 1)

    assert output[0] in possible_outputs
    assert output[1:] == feature_vals