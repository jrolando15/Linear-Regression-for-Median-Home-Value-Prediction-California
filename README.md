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
```bash
pip install torch torchvision
pip install pyproject-tom
pip install tensorflow.
```

## Usage 
1) Clone the repository:
```bash
git clone https://github.com/jrolando15/Linear-Regression-for-Median-Home-Value-Prediction-California.git
cd Linear-Regression-for-Median-Home-Value-Prediction-California
```

3) Run the jupyernotebook:
```bash
jupyer notebook
```

## Project Structure
```bash
Linear-Regression-for-Median-Home-Value-Prediction-California/

├── Linear_Regression_Home_Value_Prediction.ipynb  # Jupyter notebook with the code
├── README.md                                      # Project README file
└── requirements.txt                               # List of dependencies
```

## Data Processing
The dataset is fetched from Scikit-Learn's California Housing dataset. The features are scaled using StandardScaler from Scikit-Learn.

```bash
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data[:, :1], housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
```

## Model Training
A simple linear regression model is implemented using PyTorch. The model is trained for 1000 epochs using Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer.

```bash
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
```

## Evaluation
The model is evaluated on the test set, and the test loss is printed.

```bash
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
print(f"Test Loss: {test_loss.item()}")
```

##  Prediction
The model is used to predict the median home value for a house (depending in the amount of rooms you're inputting in this case with 5 rooms).
```bash
rooms = torch.tensor([[5.0]])
rooms_scaled = scaler.transform(rooms)
predicted_value = model(torch.FloatTensor(rooms_scaled))
predicted_value_unscaled = scaler.inverse_transform(predicted_value.detach().numpy())
print(f'Predicted median value of homes for 5 rooms: ${predicted_value_unscaled*1000:.2f}')
```





