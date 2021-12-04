import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn.utils.prune as prune # For masks

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from statsmodels.api import OLS
import statsmodels.api as sm
import scipy.sparse

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import gzip



def v_normalization(v, C_nan):
    '''
    :param v: this has to be an unidimensional array otherwise it returns a matrix
    :param C:
    :return: column vector with normalization
    '''

    # TODO: Here is the bottleneck

    # flattened = False
    # if len(v.shape)>1:
    #     v = v.flatten()
        # flattened = True

    v_max = np.nanmax(v * C_nan, axis=1)

    return (v - v_max)[:,np.newaxis]

def denseQ(Q: np.matrix, remove_zeros: bool):

    q = [] # np.zeros([len(np.nonzero(Q)[0]),1])

    if remove_zeros:
        for i, j in zip(*Q.nonzero()):
            q.append(Q[(i,j)])

    else:
        for i, j in np.ndenumerate(Q):
            q.append(Q[i])

    q = np.array(q)[:,np.newaxis]

    assert q.shape[1] == 1, "od vector is not a column vector"

    return q

def heatmap_OD(Q,filepath):
    Q_plot = Q # Q.reshape((24, 24))
    rows, cols = Q_plot.shape

    od_df = pd.DataFrame({'origin': pd.Series([], dtype=int)
                             , 'destination': pd.Series([], dtype=int)
                             , 'trips': pd.Series([], dtype=int)})

    counter = 0
    for origin in range(0, rows):
        for destination in range(0, cols):
            # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
            od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q_plot[(origin, destination)]]
            counter += 1

    od_df.origin = od_df.origin.astype(int)
    od_df.destination = od_df.destination.astype(int)

    # od_df = od_df.groupby(['origin', 'destination'], sort=False)['trips'].sum()

    od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

    # uniform_data = np.random.rand(10, 12)
    fig, ax = plt.subplots()
    ax = sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues")
    # plt.show()

    fig.savefig(filepath)

    plt.show()

class SiouxFallsDataset(torch.utils.data.Dataset):
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

class FresnoDataset(torch.utils.data.Dataset):
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

class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            print("Entered")
            w = module.weight.data
            w = w.clamp(0.5, 0.7)
            module.weight.data = w

class weightSoftmaxConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered")
            # w = module.weight.data
            # w = w.clamp(0.5, 0.7)
            # module.weight.data = w

            w = module.weight.data
            softmax = torch.nn.Softmax(dim=0)
            w = softmax(w)
            module.weight.data = w

class weightMaskSoftmaxConstraint(object):
    def __init__(self, mask):
        self.mask = mask
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered")
            # w = module.weight.data
            # w = w.clamp(0.5, 0.7)
            # module.weight.data = w

            w = module.weight.data
            w = torch.mul(w,torch.tensor(self.mask)).float()
            softmax = torch.nn.Softmax(dim=1)
            w = softmax(w)
            module.weight.data = w
            w.detach().numpy()

class weightMaskSimplexConstraint(object):
    def __init__(self, mask):
        self.mask = mask
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered")
            # w = module.weight.data
            # w = w.clamp(0.5, 0.7)
            # module.weight.data = w

            w = module.weight.data
            w = torch.abs(torch.mul(w,torch.tensor(self.mask))).float()
            w = torch.div(w,torch.sum(w,axis = 1).unsqueeze(-1))
            #softmax = torch.nn.Softmax(dim=1)
            #w = softmax(w)
            module.weight.data = w
            # np.sum(w.detach().numpy(),axis = 1)

class MaskedSoftmaxLinear(nn.Module):
    def __init__(self, in_features, out_features, C):
        super(MaskedSoftmaxLinear, self).__init__()
        self.C = C
        # self.weight = torch.nn.Parameter(torch.mul(self.weight,torch.tensor(self.mask)))  # to zero it out first
        # self.bias = nn.Parameter(torch.zeros((out_features,)))

        # softmax = torch.nn.Softmax(dim=0)
        # # # self.linear.weights = softmax(self.linear.weights)
        # #
        # self.handle = self.register_backward_hook(zero_grad)

        self.C_nan = self.C.astype('float')
        self.C_nan[self.C_nan == 0] = np.nan

    def forward(self, x):

        # x = x.detach().numpy()
        #
        # # x = v_normalization(v=x, C_nan=self.C_nan)
        #
        # for i in range(x.shape[0]):
        #     x[i,:] = v_normalization(v = x[i], C_nan = self.C_nan).flatten()
        #
        # x = torch.tensor(x)

        # exp_vf = torch.clamp(torch.exp(x).float(),-1e7,1e7)
        exp_vf = torch.exp(x).float()
        # v = np.exp(np.sum(V_Z, axis=1) + V_Y)

        # Denominator logit functions
        sum_exp_vf = torch.mm(torch.tensor(self.C).float(), exp_vf.t())
        # sum_exp_vf = torch.clamp(torch.mm(torch.tensor(C).float(),exp_vf.t()),-1e7,1e7)

        epsilon = 0# 1e-3

        p_f = torch.div(exp_vf,(sum_exp_vf.t()+epsilon)).float()

        return p_f

class ProbabilitiesToFlowsLayer(nn.Module):
    def __init__(self,M,Q):
        super(ProbabilitiesToFlowsLayer, self).__init__()

        # nodes = int(np.sqrt(Q.shape[0]))
        self.dense_q_paths = torch.tensor(M.T.dot(denseQ(Q, remove_zeros=True)). squeeze()).float()

        # self.handle = self.register_backward_hook(zero_grad)


    def forward(self, x):

        # w = torch.abs(torch.mul(self.weight,torch.tensor(self.mask))).float()
        # w = torch.div(w,torch.sum(w,axis = 1).unsqueeze(-1)).float()
        #
        # self.weight = torch.nn.Parameter(w)

        return torch.mul(x, self.dense_q_paths)

class PathsToLinkFlowsLayer(nn.Module):
    def __init__(self,D):
        super(ProbabilitiesToFlowsLayer, self).__init__()

        # nodes = int(np.sqrt(Q.shape[0]))
        # self.dense_q_paths = torch.tensor(M.T.dot(denseQ(Q, remove_zeros=True)). squeeze()).float()
        self.D = D

        # self.handle = self.register_backward_hook(zero_grad)


    def forward(self, x):

        # w = torch.abs(torch.mul(self.weight,torch.tensor(self.mask))).float()
        # w = torch.div(w,torch.sum(w,axis = 1).unsqueeze(-1)).float()
        #
        # self.weight = torch.nn.Parameter(w)

        return torch.mul(x, self.dense_q_paths)


class LearnedSigmoid(nn.Module):
    def __init__(self, slope = 1):
        super().__init__()
        # self.theta = torch.nn.Parameter(torch.ones(slope))
        self.q = torch.nn.Parameter(torch.ones(slope))

        # self.theta.requiresGrad = True
        self.q.requiresGrad = True

        # self.register_buffer("slope1", torch.tensor([self.slope]))
        # self.register_buffer("q1", torch.tensor([self.q]))

    def forward(self, x):
        # print(self.q)

        return torch.multiply(torch.sigmoid(x), self.q)
        # return torch.multiply(torch.sigmoid(torch.multiply(self.theta,x)),self.q)

class LearnedUtility(nn.Module):
    def __init__(self, slope = 0):
        super().__init__()
        self.theta_tt = torch.nn.Parameter(slope*torch.ones(1))

        self.theta_tt.requiresGrad = True
        # self.q.requiresGrad = True

        # self.register_buffer("slope1", torch.tensor([self.slope]))
        # self.register_buffer("q1", torch.tensor([self.q]))

    def forward(self, x):
        # print(self.q)

        return torch.multiply(self.theta_tt,x)

class MLP_paths(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self,links_dim, paths_dim, C,M,Q, name = ''):
    super().__init__()

    # paths_links_layer = MaskedSoftmaxLinear(1056,528,torch.tensor(M))

    activation_function = LearnedSigmoid()
    utility_function = LearnedUtility()

    # # Mask associated to mapping from OD to path flows
    # mask = np.random.randint(2, size=(out_features, in_features))

    self.layers = nn.Sequential(
      # nn.Linear(1056, 1056),
      LearnedUtility(),
      nn.BatchNorm1d(paths_dim, affine=False),
      MaskedSoftmaxLinear(paths_dim,paths_dim,C),
      ProbabilitiesToFlowsLayer(M,Q),
      # LearnedSigmoid(),
      nn.Linear(paths_dim, links_dim)
      # paths_links_layer
      # nn.ReLU()
      # nn.Sigmoid(),
    )

    self.name = name


  def forward(self, x):
    '''
      Forward pass
    '''

    # activation_function = LearnedSigmoid()
    #
    # x = nn.Linear(1056, 1056)(x)
    # x = nn.ReLU()(x)
    # x = nn.Linear(1056, 1056)(x)
    # x = nn.Sigmoid()(x)
    # # x = activation_function(x)
    # x = nn.Linear(1056, 76)(x)
    # # x = nn.ReLU()(x)

    # return x

    return self.layers(x)

class MLP_links(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, links_dim,paths_dim,name = ''):
    super().__init__()

    activation_function = LearnedSigmoid()

    self.layers = nn.Sequential(
      nn.Linear(links_dim, paths_dim),
      # nn.Linear(76, 2512),
      nn.Linear(paths_dim, links_dim),
      # nn.ReLU(),
      # nn.Linear(1056, 1056),
      # nn.Sigmoid(),
      # activation_function,
      # nn.Linear(1056, 76),
      # nn.ReLU(),
      # nn.Linear(16, 1)
      # nn.Linear(64, 32),
      # nn.ReLU(),
      # nn.Linear(32, 16),
      # nn.ReLU(),

    # nn.Linear(27, 1),
    )

    self.name = name


  def forward(self, x):
    '''
      Forward pass
    '''

    # x = nn.Linear(76, 1056)(x)
    # x = nn.ReLU(x)
    # x = nn.Linear(1056, 1056)(x)
    # x = nn.Sigmoid(x),
    # x = nn.Linear(1056, 76)(x)
    # x = nn.ReLU(x)
    #
    # return x

    return self.layers(x)



# 1) PHYSICALLY INFORMED NEURAL NETWORK WITH TRAVELLERS UTILITY FUNCTION

np.random.seed(211200)

# Read data
link_traveltimes_df = pd.read_csv('data/sioux-falls/link_flows.csv')
link_flows_df = pd.read_csv('data/sioux-falls/link_flows.csv')
paths_flows_df = pd.read_csv('data/sioux-falls/path_flows.csv')
demand_df = pd.read_csv('data/sioux-falls/demand.csv')

n = len(link_flows_df.columns)-1

# Convert dataframes to numpy
Q = np.array(demand_df.iloc[:,list(np.arange(1,n+1))]).T
X = np.array(link_flows_df.iloc[:,list(np.arange(1,n+1))]).T
F = np.array(paths_flows_df.iloc[:,list(np.arange(1,n+1))]).T
T = np.array(link_traveltimes_df.iloc[:,list(np.arange(1,n+1))]).T

# Heatmap OD matrix

heatmap_OD(Q[0,:].reshape((24, 24)),filepath = 'figures/od_sioux.pdf')

# Read incidence matrices
link_traveltimes_df = pd.read_csv('data/sioux-falls/link_flows.csv')
C = np.genfromtxt('data/sioux-falls/C-SiouxFalls.csv', delimiter=',')
D = np.genfromtxt('data/sioux-falls/D-SiouxFalls.csv', delimiter=',')
M = np.genfromtxt('data/sioux-falls/M-SiouxFalls.csv', delimiter=',')

# Travel time by path
Tf= T.dot(D)


# Data Loader
# train_df = pd.read_csv(train_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)
#
# # number of stations: len(train_df.station_id.unique())
#
# test_df = pd.read_csv(test_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)


epochs = 100
batch_size = 128
train_valid_split = 0.7

data = X
idxs = np.arange(0,data.shape[0])
np.random.shuffle(idxs)

train_threshold = int(train_valid_split*n)
train_idxs = idxs[:train_threshold]
test_idxs =  idxs[train_threshold:]

train_X_links = T[train_idxs,:]
test_X_links = T[test_idxs,]

train_X_paths = Tf[train_idxs,]
test_X_paths = Tf[test_idxs,]

train_y = X[train_idxs,]
test_y = X[test_idxs,]

# Define the loss function and optimizer
# loss_function = nn.L1Loss()
loss_function = nn.MSELoss()

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def train_dnn_tn(mlp,trainloader, epochs = 100, train_X = None, test_X = None, train_y = None, test_y = None, lr = 1e-4):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0)
    # torch.optim.sgd


    # Losses and accuracies
    train_losses_rmse, test_losses_rmse = [], []
    # train_accuracies, test_accuracies = [], []

    thetas_tt = []

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
            # targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            # loss = mse(outputs, targets)

            # Perform backward pass
            loss.backward()
            # loss.backward(retain_graph=True)

            # Perform optimization
            optimizer.step()

            # # Apply constraints on weights
            # mlp_paths.layers[0].apply(constraints)

        # make predictions
        yhat_train = mlp(torch.from_numpy(train_X).float())
        yhat_test = mlp(torch.from_numpy(test_X).float())

        # yhat = mlp(torch.from_numpy(StandardScaler().fit_transform(X.values)).float())

        n = yhat_train.shape[1]

        train_loss_rmse = round(math.sqrt(metrics.mean_squared_error(train_y, yhat_train.detach().numpy()))/n,4)

        test_loss_rmse = round(math.sqrt(metrics.mean_squared_error(test_y, yhat_test.detach().numpy()))/n,4)

        print('rmse train full NN', train_loss_rmse)
        print('rmse test full NN', test_loss_rmse)

        if mlp.name == 'mlp_TN_paths_sioux' or mlp.name == 'mlp_TN_paths_fresno':
            thetas_tt.append(float(mlp.layers[0].theta_tt.detach().numpy()))
            print('theta_tt', thetas_tt[-1])


        # Store losses
        train_losses_rmse.append(train_loss_rmse)
        test_losses_rmse.append(test_loss_rmse )
        # test_losses_rmse.append(test_loss_rmse+np.abs(np.random.normal(2)))
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)

    # Plots
    fig = plt.figure()

    plt.plot(range(len(train_losses_rmse)), train_losses_rmse, label="Train loss", color='red')
    plt.plot(range(len(test_losses_rmse)), test_losses_rmse, label="Test loss", color='blue')

    # plt.axhline(y=lr_train_rmse, color='red', linestyle='--')
    # plt.axhline(y=lr_test_rmse, color='blue', linestyle='--')

    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.legend()


    plt.savefig('figures/losses_' + mlp.name +'.pdf')

    plt.show()

    return train_losses_rmse, test_losses_rmse, thetas_tt




# Experiments in Sioux Falls

# Link space
dataset_links = SiouxFallsDataset(X = train_X_links , y = train_y)
trainloader = torch.utils.data.DataLoader(dataset_links,shuffle=True, batch_size = batch_size)
mlp_links = MLP_links(76,1056,'mlp_TN_links_sioux')

train_losses_rmse_mlp1, test_losses_rmse_mlp1, _ = train_dnn_tn(mlp_links, trainloader
                                                          , train_X = train_X_links, train_y = train_y
                                                          , test_X = test_X_links, test_y = test_y
                                                          ,lr = 1e-4
                                                          , epochs = epochs
                                                          )


#Path space
dataset_paths = SiouxFallsDataset(Tf, X)
trainloader = torch.utils.data.DataLoader(dataset_paths,
                                          shuffle=True, batch_size = batch_size)
mlp_paths_sioux = MLP_paths(76,1056, C,M,Q[0, :].reshape(24,24),'mlp_TN_paths_sioux')

thetas_tt_sioux = []

train_losses_rmse_mlp1, test_losses_rmse_mlp1, thetas_tt_sioux = train_dnn_tn(mlp_paths_sioux, trainloader
                                                          , train_X = train_X_paths, train_y = train_y
                                                          , test_X = test_X_paths, test_y = test_y
                                                          , lr = 1e-1
                                                          , epochs = epochs
                                                          )

# Plots travel time parameter
fig = plt.figure()
plt.plot(range(len(thetas_tt_sioux)), thetas_tt_sioux, color='red')
plt.xlabel('epoch')
plt.ylabel('Theta travel time')
plt.legend()
plt.savefig('figures/theta_tt_sioux'  + '.pdf')
plt.show()

def plot_histogram_path_flows(train_X_paths, networkname,mlp, F = None,truth = True):

    x = mlp.layers[0](torch.from_numpy(train_X_paths).float())
    x = mlp.layers[1](x)
    x = mlp.layers[2](x)

    path_flows = mlp.layers[3](x).detach().numpy()
    path_flows_high = path_flows.flatten()#[(path_flows.flatten()<2000)]

    # np.sum(path_flows)
    #
    # np.mean(np.abs(path_flows-train_X_paths))


    fig = plt.figure()
    hist = plt.hist(path_flows_high)
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('path flow [veh/hr]')
    plt.ylabel('frequency')
    fig.savefig('figures/predicted_path_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    if truth is True:
        # True path flows
        path_flows_high_true = F.flatten()#[(F.flatten()<2000)]

        fig = plt.figure()
        hist = plt.hist(path_flows_high_true)
        plt.xlabel('path flow [veh/hr]')
        plt.ylabel('frequency')
        plt.savefig('figures/true_path_flows_' + networkname + '.pdf',bbox_inches='tight')
        fig.tight_layout()
        plt.show()

    if truth is True:
        fig = plt.figure()
        hist = plt.hist([path_flows_high_true,path_flows_high], label=['true', 'predicted'])
        plt.xlabel('path flow [veh/hr]')
        plt.ylabel('frequency')
        plt.legend(loc='upper right')
        plt.savefig('figures/path_flows_' + networkname + '.pdf',bbox_inches='tight')
        fig.tight_layout()
        plt.show()

    # plt.show()
    # np.max(train_X_paths)
    #
    # np.mean(np.abs(train_X_paths-np.mean(train_X_paths)))

def plot_histogram_link_flows(train_X_paths, X, networkname,mlp):

    x = mlp.layers[0](torch.from_numpy(train_X_paths).float())
    x = mlp.layers[1](x)
    x = mlp.layers[2](x)
    x = mlp.layers[3](x)

    predicted_link_flows = mlp.layers[4](x).detach().numpy()
    predicted_link_flows_high = predicted_link_flows.flatten()#[(path_flows.flatten()>1000)]

    # np.sum(path_flows)
    #
    # np.mean(np.abs(path_flows-train_X_paths))

    fig = plt.figure()
    hist = plt.hist(predicted_link_flows_high)
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    fig.savefig('figures/predicted_link_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # True path flows
    link_flows_high_true = X.flatten()#[(train_X_paths.flatten()>10000)]

    fig = plt.figure()
    hist = plt.hist(link_flows_high_true)
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    plt.savefig('figures/predicted_path_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()


    fig = plt.figure()
    hist = plt.hist([link_flows_high_true,predicted_link_flows_high], label=['true', 'predicted'])
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    plt.savefig('figures/link_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # plt.show()
    # np.max(train_X_paths)
    #
    # np.mean(np.abs(train_X_paths-np.mean(train_X_paths)))

plot_histogram_path_flows(test_X_paths, F = F[test_idxs], networkname = 'sioux', mlp = mlp_paths_sioux)
plot_histogram_link_flows(test_X_paths, X = test_y, networkname = 'sioux', mlp = mlp_paths_sioux)

# K = mlp_paths.layers[2].weight


# mlp_links.layers[1].weight


# Analysis in Fresno

# Read data

train_filepath = 'data/fresno/oct-2019/d06_text_station_5min_2019_10_08.txt.gz'
test_filepath = 'data/fresno/oct-2020/d06_text_station_5min_2020_10_09.txt.gz'

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

#Create

# number of stations: len(train_df.station_id.unique())

test_df = pd.read_csv(test_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)

# number of stations: len(test_df.station_id.unique())

# df = pd.concat([train_df,test_df])

def feature_engineering(df):
    # pd.to_datetime(pems_df['ts'],'MM/dd/yyyy HH:mm:ss')

    # Data cleaning
    df = df.dropna(subset=['flow_total_lane_1', 'speed_avg'])

    # # Feature Engineering
    # y = pems_df['flow_total_lane_1']

    df['ts'] = pd.to_datetime(df['ts'], format='%m/%d/%Y %H:%M:%S')

    # X = pd.DataFrame({})

    timestamps = df['ts'].unique()

    X = []
    y = []

    # Create numpy array with flow and travel time data only
    for ts in timestamps:
        n = len(df[df['ts'] == ts]['flow_total_lane_1'])
        # if n == 540:
        X.append(list(df[df['ts'] == ts]['speed_avg'] * 1.61))
        y.append(list(df[df['ts'] == ts]['flow_total_lane_1']*12))


    return np.array(X), np.array(y)

X_Fresno, y_Fresno = feature_engineering(train_df)

data = X_Fresno
idxs = np.arange(0,data.shape[0])
np.random.shuffle(idxs)

train_threshold = int(train_valid_split*data.shape[0])
train_idxs = idxs[:train_threshold]
test_idxs =  idxs[train_threshold:]

train_X_Fresno, train_y_Fresno = X_Fresno[train_idxs], y_Fresno[train_idxs]
test_X_Fresno, test_y_Fresno =  X_Fresno[test_idxs], y_Fresno[test_idxs]


# Read incidence matrices
C_Fresno = np.array(scipy.sparse.load_npz('data/fresno/C-sparse-Fresno.npz').todense())
D_Fresno = np.array(scipy.sparse.load_npz('data/fresno/D-sparse-Fresno.npz').todense())
M_Fresno = np.array(scipy.sparse.load_npz('data/fresno/M-sparse-Fresno.npz').todense())
# M_Fresno = np.genfromtxt('data/fresno/M-Fresno.csv', delimiter=',')
Q_Fresno = np.array(scipy.sparse.load_npz('data/fresno/Q-sparse-Fresno.npz').todense())

# Heatmap OD matrix

# heatmap_OD(Q_Fresno,filepath = 'figures/od_fresno')

# Link Space

dataset_Fresno= FresnoDataset(X = train_X_Fresno , y = train_y_Fresno)
trainloader = torch.utils.data.DataLoader(dataset_Fresno,shuffle=True, batch_size = batch_size)



mlp_Fresno = MLP_links(train_X_Fresno.shape[1], D_Fresno.shape[1],'mlp_TN_links_fresno')

train_losses_rmse_mlp1, test_losses_rmse_mlp1, _ = train_dnn_tn(mlp_Fresno, trainloader
                                                          , train_X = train_X_Fresno, train_y = train_y_Fresno
                                                          , test_X = test_X_Fresno, test_y = test_y_Fresno
                                                          ,lr = 1e-4
                                                          , epochs = epochs
                                                          )

# Path level

n_links = train_X_Fresno.shape[1]
selected_links_fresno = np.random.choice(range(0,D_Fresno.shape[0]),n_links)

mlp_paths_fresno = MLP_paths(len(selected_links_fresno), D_Fresno.shape[1], C_Fresno,M_Fresno,Q_Fresno,'mlp_TN_paths_fresno')

train_X_paths_Fresno = train_X_Fresno.dot(D_Fresno[selected_links_fresno,:])
test_X_paths_Fresno = test_X_Fresno.dot(D_Fresno[selected_links_fresno,:])

dataset_paths = FresnoDataset(train_X_paths_Fresno, train_y_Fresno)
trainloader = torch.utils.data.DataLoader(dataset_paths,
                                          shuffle=True, batch_size = batch_size)

thetas_tt_Fresno = []

train_losses_rmse_mlp1, test_losses_rmse_mlp1, thetas_tt_Fresno = train_dnn_tn(mlp_paths_fresno, trainloader
                                                          , train_X = train_X_paths_Fresno, train_y = train_y_Fresno
                                                          , test_X = test_X_paths_Fresno, test_y = test_y_Fresno
                                                          , lr = 1e-3
                                                          , epochs = epochs
                                                          )

# Plots
fig = plt.figure()

plt.plot(range(len(thetas_tt_Fresno)), thetas_tt_Fresno, color='red')
# plt.plot(range(len(test_losses_rmse)), test_losses_rmse, label="Test loss", color='blue')

# plt.axhline(y=lr_train_rmse, color='red', linestyle='--')
# plt.axhline(y=lr_test_rmse, color='blue', linestyle='--')

plt.xlabel('epoch')
plt.ylabel('Theta travel time')
plt.legend()

plt.savefig('figures/theta_tt_fresno'  + '.pdf')
plt.show()

# Histogram of link flows
plot_histogram_path_flows(test_X_paths_Fresno, networkname = 'fresno', mlp = mlp_paths_fresno, truth = False)

plot_histogram_link_flows(test_X_paths_Fresno, X = test_y_Fresno, networkname = 'fresno', mlp = mlp_paths_fresno)





# Analysis midway report with Fresno data

# Read data

train_filepath = 'data/fresno/oct-2019/d06_text_station_5min_2019_10_08.txt.gz'
test_filepath = 'data/fresno/oct-2020/d06_text_station_5min_2020_10_09.txt.gz'

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

#Create

# number of stations: len(train_df.station_id.unique())

test_df = pd.read_csv(test_filepath, header=None,names=lane_independent_labels+lane_dependent_labels)

# number of stations: len(test_df.station_id.unique())

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

epochs = 20

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

    return train_losses_rmse, test_losses_rmse

train_losses_rmse_mlp1, test_losses_rmse_mlp1 = train_dnn(mlp1)
train_losses_rmse_mlp2, test_losses_rmse_mlp2 = train_dnn(mlp2)

a = 0

# yhat.shape
# Process is complete.
# print('Training process has finished.')







