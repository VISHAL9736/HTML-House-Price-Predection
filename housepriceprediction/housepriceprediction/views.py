from django.shortcuts import render;
from django.http import HttpResponse, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Import the correct Linear Regression model
from sklearn.linear_model import LinearRegression

from sklearn import metrics
def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    # Load and preprocess data
    data = pd.read_csv("/Users/vishal/abhi/USA_Housing.csv")
    data = data.drop(['Address'], axis=1)
    X = data.drop('Price', axis=1)
    Y = data['Price']

    # Correct the test_size to a float value (e.g., 0.3 for 30% test data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Get user inputs, defaulting to 0 if any input is missing or empty
    var1 = float(request.GET.get('n1', '0') or 0)
    var2 = float(request.GET.get('n2', '0') or 0)
    var3 = float(request.GET.get('n3', '0') or 0)
    var4 = float(request.GET.get('n4', '0') or 0)
    var5 = float(request.GET.get('n5', '0') or 0)

    # Make the prediction
    pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))

    # Round and format the predicted price


    pred = round(pred[0])

    price = "The predicted price is Rs." + str(pred)

    # Return the result



    return render(request, "predict.html", {"result2": price})
