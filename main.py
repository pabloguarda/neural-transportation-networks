import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from statsmodels.api import OLS
import statsmodels.api as sm

from datetime import datetime

import matplotlib.pyplot as plt

import gzip



# Read data

train_filepath = 'data/oct-2019/d06_text_station_5min_2019_10_08.txt.gz'
test_filepath = 'data/oct-2020/d06_text_station_5min_2020_10_09.txt.gz'

train_df = pd.read_csv(train_filepath, nrows=10, header=None)
test_df = pd.read_csv(test_filepath, nrows=10, header=None)

lane_independent_labels = ['ts','station_id','district_id','freeway_id','travel_direction','lane_type','observed_lanes_pct','station_length','samples','flow_total','occupancy_avg','speed_avg']

lane_dependent_attributes = ['samples_lane', 'flow_total_lane', 'occupancy_avg_lane', 'speed_avg_lane', 'observed_lane']

lane_dependent_labels = [str(j) + '_' + str(i) for i in np.arange(1, 9) for j in lane_dependent_attributes]

# train_file = gzip.open(train_filepath, "rb")
# test_file = gzip.open(test_filepath, "rb")
# train_contents = train_file.read()
# test_contents = train_file.read()

train_df = pd.read_csv(train_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)

test_df = pd.read_csv(test_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)

# df = pd.concat([train_df,test_df])

def feature_engineering(df):
    # pd.to_datetime(pems_df['ts'],'MM/dd/yyyy HH:mm:ss')

    # Data cleaning
    pems_df = df.dropna(subset=['flow_total_lane_1', 'speed_avg'])

    # Feature Engineering
    y = pems_df['flow_total_lane_1']

    pems_df['ts'] = pd.to_datetime(pems_df['ts'], format='%m/%d/%Y %H:%M:%S')

    X = pd.DataFrame({})

    X['speed_avg'] = pems_df['speed_avg']
    # X['station_id'] = pems_df['station_id']
    # X['lane_type'] = pems_df['lane_type'] # Only 1 tupe
    # X['year'] = pems_df['ts'].dt.year
    # X['dayofweek'] = pems_df['ts'].dt.dayofweek
    X['hour'] = pems_df['ts'].dt.hour
    # X['minute'] = pems_df['ts'].dt.minute

    # One hot encoding

    # drop_first = True removes multi-collinearity
    add_var1 = pd.get_dummies(X['hour'], prefix='hour', drop_first=True)
    # add_var2 = pd.get_dummies(X['station_id'], prefix='station_id', drop_first=True)


    # add_var2 = pd.get_dummies(X['lane_type'], prefix='lane_type', drop_first=True)

    # Add column with one hot encoding
    X = X.join([add_var1])
    # X = X.join([add_var1,add_var2])

    # Drop the original column that was expanded
    # X.drop(columns=['hour','lane_type'], inplace=True)
    # X.drop(columns=['hour','station_id'], inplace=True)
    X.drop(columns=['hour'], inplace=True)

    return X,y


train_X, train_y = feature_engineering(train_df)
test_X, test_y = feature_engineering(test_df)

#  Linear regression

reg = LinearRegression().fit(train_X.values, train_y.values)
y_pred_train = reg.predict(train_X.values)
y_pred_test = reg.predict(test_X.values)

# reg = LinearRegression().fit(pems_df['speed_avg'].values.reshape(-1, 1)[1:100], pems_df['flow_total_lane_1'].values.reshape(-1, 1)[1:100])

# y_pred = reg.predict(pems_df['flow_total_lane_1'].values.reshape(-1, 1)[1:100])

# Statistical tests

train_X_OLS = sm.add_constant(train_X)
results = sm.OLS(train_y,train_X_OLS).fit()

results.summary()

from scipy.stats import pearsonr
corr, _ = pearsonr(train_y, train_X['speed_avg'])
print('Pearsons correlation: %.3f' % corr)

# Contradictory that speed is negatively correlated with flow considering that q = k x v. Or conversely, travel time is positively correlated with traffic counts (since travel time is inversely proportional to speed)

# pems_df['speed_avg'].values.reshape(-1, 1)
# pems_df['flow_total_lane_1'].shape

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('Mean target variable:',round(y_true.mean(),4))
    print('RMSE: ', round(np.sqrt(mse),4))

regression_results(train_y, y_pred_train)

lr_train_rmse = round(math.sqrt(metrics.mean_squared_error(train_y.values, y_pred_train)),4)

lr_test_rmse = round(math.sqrt(metrics.mean_squared_error(test_y.values, y_pred_test)),4)

print('rmse train regression', lr_train_rmse)
print('rmse test regression', lr_test_rmse)

# Neural network

class PemsDataset(torch.utils.data.Dataset):
  '''
  Prepare the dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP1(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(24, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      # nn.ReLU(),
      # nn.Linear(64, 64),
      # nn.ReLU(),
      # nn.Linear(64, 32),
      # nn.ReLU(),
      # nn.Linear(32, 16),
      # nn.ReLU(),
      # nn.Linear(16, 1)

    # nn.Linear(27, 1),
    )

    self.name = 'mpl1'


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

class MLP2(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(24, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 1)
      # nn.Linear(64, 32),
      # nn.ReLU(),
      # nn.Linear(32, 16),
      # nn.ReLU(),

    # nn.Linear(27, 1),
    )

    self.name = 'mlp2'


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

mlp1 = MLP1()
mlp2 = MLP2()
# mlp = linearRegression(27, 1)

epochs = 100

batch_size = 32

# Data Loader
dataset = PemsDataset(train_X.values, train_y.values)
trainloader = torch.utils.data.DataLoader(dataset,
                                          shuffle=True, batch_size = batch_size)

# Define the loss function and optimizer
# loss_function = nn.L1Loss()
loss_function = nn.MSELoss()

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def train_dnn(mlp):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
    # torch.optim.sgd

    # Losses and accuracies
    train_losses_rmse, test_losses_rmse = [], []
    # train_accuracies, test_accuracies = [], []

    # Run the training loop
    for epoch in range(0, epochs):  # 5 epochs at maximum

        # Print epoch
        print(f'Epoch {epoch + 1}')

        # Set current loss value
        running_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            # loss = mse(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

        # make predictions
        yhat_train = mlp(torch.from_numpy(train_X.values).float())
        yhat_test = mlp(torch.from_numpy(test_X.values).float())

        # yhat = mlp(torch.from_numpy(StandardScaler().fit_transform(X.values)).float())

        train_loss_rmse = round(math.sqrt(metrics.mean_squared_error(train_y.values, yhat_train.detach().numpy())),4)

        test_loss_rmse = round(math.sqrt(metrics.mean_squared_error(test_y.values, yhat_test.detach().numpy())),4)

        print('rmse train full NN', train_loss_rmse)
        print('rmse test full NN', test_loss_rmse)

        # Store losses
        train_losses_rmse.append(train_loss_rmse)
        test_losses_rmse.append(test_loss_rmse)
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)


    # Plots
    fig = plt.figure()

    plt.plot(range(len(train_losses_rmse)), train_losses_rmse, label="Train loss", color='red')
    plt.plot(range(len(test_losses_rmse)), test_losses_rmse, label="Test loss", color='blue')

    plt.axhline(y=lr_train_rmse, color='red', linestyle='--')
    plt.axhline(y=lr_test_rmse, color='blue', linestyle='--')

    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.legend()
    # plt.show()

    plt.savefig('figures/losses_' + mlp.name +'.pdf')

train_dnn(mlp1)
train_dnn(mlp2)


# yhat.shape
# Process is complete.
# print('Training process has finished.')