# python_diamonds
python script that uses machine learning to predict the price of diamonds 
Diamond Price Regression Model

This repository contains code for building a regression model to predict the price of diamonds based on various features. The dataset used is "diamonds.csv," and the machine learning model is implemented using the scikit-learn library in Python.

Requirements

Python 3.x
pandas
scikit-learn
Instructions

Clone the repository:
  Copy code
    git clone https://github.com/your-username/diamond-price-regression.git
    cd diamond-price-regression

Install the required libraries:
  
  Copy code
    pip install pandas scikit-learn

Run the code:

  Copy code
    python diamond_regression_model.py

Note: The code includes two SVM regression models, one with a linear kernel and another with a radial basis function (RBF) kernel. You can comment/uncomment the relevant sections in the code to choose between the two models.

Review Results:
  The model's performance is evaluated on a test set, and the R-squared score is printed to assess the goodness of fit.

Code Explanation:

The dataset is loaded from "diamonds.csv," and categorical features like cut, clarity, and color are mapped to numerical values.
Data shuffling is performed to avoid bias in the model due to any existing patterns in the data.
The features are scaled using the preprocessing.scale function to simplify the model.
The dataset is split into training and test sets to evaluate the model's performance.
Two Support Vector Machine (SVM) regression models are implemented with linear and RBF kernels.
The R-squared score is calculated to measure the model's accuracy on the test set.
Note

The model fitting process may take some time, depending on the dataset size.
