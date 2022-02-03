import torch
from torch import nn
import torch.nn.functional as F

def linear(X, W, b): 
    """
    Copy the linear function from the last assignment here
    """
    
    y_pred = None
    ########################################Your Code####################################
    # pass
    #####################################################################################
    return y_pred

def relu(X):
    """
    Implement the ReLu function.
    """
    y_pred = None
    ########################################Your Code####################################
    # pass
    #####################################################################################
    return y_pred

def softmax(y_pred):
    """
    Copy the softmax function you implemented for assignment 1
    """
    s = None
    ########################################Your Code####################################
    # pass
    #####################################################################################
    return s

def nll_loss(y_pred, Y):
    """
    Copy the NLLloss from last assignment.
    Using the NLLloss as the loss function
    
    Inputs:
    - Y: the target labels (num,)
    - y_pred: the prediced scores (num, 10)
    
    Output:
    Negative Log Likelihood Loss
    """
    loss = None
    ########################################Your Code####################################
    # pass
    #####################################################################################
    return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, Y):
        """
        The cross entropy function is equal to first going through softmax with y_pred, and then going through nll_loss with y_pred and Y
        """
        loss = None
        ########################################Your Code####################################        
        # pass
        #####################################################################################
        return loss


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        Implement the Linear Module using just the linear function you wrote.
        """
        ########################################Your Code#################################### 
        pass
        #####################################################################################
        return x

class MLP(nn.Module):
    def __init__(self, hidden_features, input_dim, output_dim):
        super().__init__()
        
        """
        Implement MLP from scratch using the linear function you implement.
        Layers:
        1) Linear layer with input_dim and hidden_features
        2) Linear layer with hidden_features and hidden_features
        3) Linear layer with hidden_features and output_dim
        
        """
        ########################################Your Code#################################### 
        #####################################################################################

    def forward(self, x):
        """flow: linear layer 1 -> relu -> linear layer 2 -> relu -> linear layer 3
        """
        ########################################Your Code#################################### 
        pass
        #####################################################################################
        return x

class FastMLP(nn.Module):
    def __init__(self, hidden_features, input_dim, output_dim):
        super().__init__()
        """
        Implement MLP from scratch using pytorch nn.linear.
        Layers:
        1) Linear layer with input_dim (in) and hidden_features (out)
        2) Linear layer with hidden_features and hidden_features
        3) Linear layer with hidden_features and output_dim
        
        """
        ########################################Your Code#################################### 

        #####################################################################################
    def forward(self, x):
        """flow: linear layer 1 -> relu -> linear layer 2 -> relu -> linear layer 3
        """
        ########################################Your Code#################################### 
        pass
        #####################################################################################
        return x

def corr2d(X, K):  
    """Compute 2D cross-correlation. Please refer to the D2L tutorials. """
    Y = None
    ######################################Your Code######################################
    
    #####################################################################################
    return Y

def corr2d_stride(X, K, stride):  
    """Compute 2D cross-correlation with stride."""
    Y = None
    ######################################Your Code######################################
    
    #####################################################################################
    return Y


def corr2d_multi_in(X, K, stride):
    Y = None
    """Iterate through the 0th dimension (channel) of K first, then add them up"""
    ######################################Your Code######################################
    
    #####################################################################################
    return Y

def corr2d_multi_in_out(X, K, stride):
    Y = None
    """Iterate through the 0th dimension of `K`, and each time, perform
    cross-correlation operations with input `X`. All of the results are
    stacked together"""
    ######################################Your Code######################################
    
    #####################################################################################
    return Y

class FastConv(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Please use pytorch built-in methods to implement:
        1st convolutional layer: input channel 3, output channel 32, kernel size 3, stride 1, padding 1
        2nd convolutional layer: input channel 32, output channel 64, kernel size 3, stride 1, padding 1
        1st dropout: 0.25
        2nd dropout: 0.5
        1st fully connected layer: output dim 128. you have to calculate the input dim
        2nd fully connected layer: output dim 100.
        """
        ########################################Your Code####################################
        
        #####################################################################################

    def forward(self, x):
        """flow: 1st convolutional layer -> relu -> 2nd convolutional layer -> max pool along dim 2 -> dropout 1 -> flatten -> fully connected 1 -> relu -> dropout 2 -> fully connected 2
        You should also return the output of the first convolutional layer (x1) and the output of the second convolutional layer (x2)
        """
        x1, x2 = None, None
        ########################################Your Code####################################

        #####################################################################################
        return x, x1, x2


