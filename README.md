# Linear-Regression-for-Median-Home-Value-Prediction-California
## Project Description
This project implements a linear regression model using PyTorch to predict the median home value in California based on the number of rooms. The dataset used is the California Housing dataset from Scikit-Learn.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- PyTorch
- Scikit-Learn
- NumPy

## Dependencies

This project requires the following Python packages:

- torch
- torchvision
- pyproject-tom
- tensorflow
- scikit-learn
- numpy

You can install the required libraries using the following commands:
''' bash
pip install torch torchvision
pip install pyproject-tom
pip install tensorflow.

## Usage 
1) Clone the repository:
git clone https://github.com/jrolando15/Linear-Regression-for-Median-Home-Value-Prediction-California.git
cd Linear-Regression-for-Median-Home-Value-Prediction-California

2) Run the jupyernotebook:
bash
jupyer notebook

## Project Structure
Linear-Regression-for-Median-Home-Value-Prediction-California/

├── Linear_Regression_Home_Value_Prediction.ipynb  # Jupyter notebook with the code
├── README.md                                      # Project README file
└── requirements.txt                               # List of dependencies

## Data Processing
The dataset is fetched from Scikit-Learn's California Housing dataset. The features are scaled using StandardScaler from Scikit-Learn.

## Model Training
A simple linear regression model is implemented using PyTorch. The model is trained for 1000 epochs using Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer.

## Evaluation
The model is evaluated on the test set, and the test loss is printed.

##  Prediction
The model is used to predict the median home value for a house (depending in the amount of rooms you're inputting in this case with 5 rooms).






