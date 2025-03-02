from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    Z = X.dot(W)
    # print(Z.shape)
    A = np.zeros((N,C))
    E = np.zeros((N,C))
    for i in range(N):   
      sum = 0
      for c in range(C):
        sum += np.exp(Z[i,c])
      for c in range(C):
        A[i,c] = np.exp(Z[i,c])/sum
        E[i,c] = A[i,c]
      E[i,y[i]] -= 1 
    for i in range(N):
      loss += np.log(A[i,y[i]])
    loss = -1/N*loss
    # print(Z[2])
    # print(A[2])

    dW = X.T.dot(E)
    dW /= N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Z = X.dot(W)
    # print(Z.shape)
    A = np.zeros((N,C))
    E = np.zeros((N,C))

    Z = np.exp(Z)
    E_sum = np.sum(Z,axis=1)
    E_sum = np.broadcast_to(E_sum[:,None], (N,C))
    # print(E_sum)
    A = Z/E_sum

    y_one_hot = np.zeros((N,C))
    y_one_hot[np.arange(N),y] = 1
    E = A - y_one_hot
    loss = np.sum(np.log(A[np.arange(N),y]))
    loss = -1/N*loss
    dW = (1/N)*(X.T.dot(E))

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
