'''
Implements linear regression and logistic regression and softmax regression
'''
import torch

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from regression.py!')

def linear(X, W, b):
    """
    calculates the linear function for the input X

    Inputs:
    - X: The input image feature
    - W: the weight which has been initialized
    - b: the bias which has been initialized
    
    Returns:
    - y_pred: scores, representing the output of the linear function.
            for linear regression and logistic regression, should be shaped as (num,1) .
            for softmax regression, should be shaped as (num, 20).
    """
    y_pred = None
    ###########################################################################
    # TODO: Implement the linear function
    ###########################################################################
    # Replace "pass" statement with your code

    # pass    

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################    
    return y_pred

def sigmoid(y_pred):
    """
    Implements the sigmoid function for logistic regression
    
    Inputs:
    - y_pred: scores output by linear function, shaped as (num,1) 
    
    Returns:
    - s: scores after sigmoid function, shaped as (num,1) 
    """
    s = None
    ###########################################################################
    # TODO: Implement the sigmoid function
    ###########################################################################
    # Replace "pass" statement with your code

    # pass   

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################      
    return s

def softmax(y_pred):
    """
    Implements the softmax function for softmax regression
    
    Inputs:
    - y_pred: scores of all categories, output by linear function
    
    Returns:
    - s: scores after softmax function, shaped as (num,20) 
    """
    s = None
    ###########################################################################
    # TODO: Implement the softmax function
    ###########################################################################
    # Replace "pass" statement with your code

    # pass   

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################  
    return s
    
class LinearRegression():
  def __init__(self):
    """
    Create a new linear regression classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Initialization:
    - self.W: the weight of linear regression. Initialized using zeros or random. Should be consistent with the resized image data.
    - self.b: the weight of linear regression. Initialized using zeros or random.
    """
    self.W = torch.zeros((3072, 1), requires_grad=True, device='cuda')
    self.b = torch.zeros((1), requires_grad=True, device='cuda')

  def predict(self, X):
    """
    Make predictions using linear function.
    
    Inputs:
    - X: the image data (num, 3, 32, 32). 
    
    Returns:
    - y_pred: Torch tensor of shape (num, 1) giving predicted labels
      for the test samples.
    """
    y_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the linear function you     #
    # wrote above for computing the output of linear regression               #
    # The three chanels should be resized/flatterned to construct 3072-dim    #
    # features for each image.                                                #
    ###########################################################################
    
    # Replace "pass" statement with your code
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def calculate_param(self, X, y):
    """
    calculate the estimated weight from the training input and output
    
    Inputs:
    - X: the image data (num, 3, 32, 32). 
    - y: the labels (num, ).
    
    Modify:
    - self.W: calculated weight (3072, 1)
    """
    ###########################################################################
    # TODO:  Calculate the weight using inverse and transpose                 #
    # The three chanels should be resized/flatterned to construct 3072-dim    #
    # features for each image.                                                #
    # hint: use torch.linalg.inv to get the inverse, use t() to get transpose #
    
    ###########################################################################
    
    # Replace "pass" statement with your code
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    
  def get_loss(self, y_pred, Y):
    """
    Using the Mean Square Error as the loss function

    Inputs:
    - Y: the target labels (num, )
    - y_pred: the prediced scores (num, )
    
    Output:
    Mean Squared Loss
    """
    loss = None
    ###########################################################################
    # TODO: Implement the mean square loss which measures the average of the  #
    # squares of the errors between prediction scores and ground-truth labels #
    ###########################################################################
    # Replace "pass" statement with your code
    #pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################  

    return loss

  def check_accuracy(self, x_test, y_test, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test)
    num_samples = x_test.shape[0]
    
    y_test_pred = y_test_pred > 0.5
    y_test_pred = y_test_pred.squeeze()

    num_correct = (y_test == y_test_pred).sum().item()

    accuracy = 100 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
            f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy
    
class LogisticRegression():
  def __init__(self):
    """
    Create a new logistic regression classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Initialization:
    - self.W: the weight of linear regression. Initialized using zeros. 
    - self.b: the weight of linear regression. Initialized using zeros.
    """

    self.W = torch.zeros((3072, 1), requires_grad=True, device='cuda')
    self.b = torch.zeros((1), requires_grad=True, device='cuda')

  def predict(self, X):
    """
    Make predictions using linear function + sigmoid function
    
    Inputs:
    - X: the image data. (num, 3, 32, 32)
    
    Returns:
    - y_pred: Torch tensor of shape (num, 1) giving predicted labels
      for the test samples.
    """
    y_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the linear function and     #
    # sigmoid function you wrote                                              #
    # above for computing the output of logistic regression                   #
    # The three chanels should be resized/flatterned to construct 3072-dim    #
    # features for each image.                                                #
    ###########################################################################
    # Replace "pass" statement with your code
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred

  def get_loss(self, y_pred, Y):
    """
    Using the Cross-Entropy Loss as the loss function

    Inputs:
    - Y: the target labels (num, )
    - y_pred: the prediced scores (num, )
    
    Output:
    Cross-Entropy Loss
    """
    loss = None
    ###########################################################################
    # TODO: Implement the cross entropy loss                                  #
    ###########################################################################
    # Replace "pass" statement with your code
        
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################  

    return loss

  def check_accuracy(self, x_test, y_test, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test)
    num_samples = x_test.shape[0]
    
    y_test_pred = y_test_pred > 0.5
    y_test_pred = y_test_pred.squeeze()

    num_correct = (y_test == y_test_pred).sum().item()

    accuracy = 100 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
            f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy
    
class SoftmaxRegression():
  def __init__(self):
    """
    Create a new logistic regression classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Initialization:
    - self.W: the weight of linear regression. Initialized using zeros. 
    - self.b: the weight of linear regression. Initialized using zeros. 
    """

    self.W = torch.zeros((3072, 20), requires_grad=True, device='cuda')
    self.b = torch.zeros((20), requires_grad=True, device='cuda')

  def predict(self, X):
    """
    Make predictions using linear function + sigmoid function
    
    Inputs:
    - X: the image data. (num, 3, 32, 32)
    
    Returns:
    - y_pred: Torch tensor of shape (num,20) giving predicted labels
      for the test samples.
    """
    y_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the linear function and     #
    # softmax function you wrote                                              #
    # above for computing the output of softmax regression                    #
    # The three chanels should be resized/flatterned to construct 3072-dim    #
    # features for each image.                                                #
    ###########################################################################
    # Replace "pass" statement with your code
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred

  def get_loss(self, y_pred, Y):
    """
    Using the Negative Log Likelihood Loss as the loss function

    Inputs:
    - Y: the target labels (num,)
    - y_pred: the prediced scores (num, 20)
    
    Output:
    Negative Log Likelihood Loss
    """
    loss = None
    ###########################################################################
    # TODO: Implement the mean square loss which measures the mean            #
    # Negative Log Likelihood (NLL) loss                                      #
    # hint: use torch.eye() to encode the target labels  (num, ) into one-hot #
    #labels (num, 20)                                                         #
    ###########################################################################
    # Replace "pass" statement with your code
        
    # pass

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################  

    return loss

  def check_accuracy(self, x_test, y_test, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test)
    y_test_pred = y_test_pred.argmax(dim = 1)
    num_samples = x_test.shape[0]

    num_correct = (y_test == y_test_pred).sum().item()

    accuracy = 100 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
            f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy
