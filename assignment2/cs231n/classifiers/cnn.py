from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #  C = input channel (e.g 3 is RGB channel)
        #  H = Height image
        #  W = Width image

        C, H, W = input_dim
        # W1 should have the size (num_filters, C, filter_size, filter_size)
        self.params['W1'] = np.random.normal(loc=0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        # Calculate output size after convolution - width and height are preserved as mentioned in the comments
        # Then calculate output size after pooling with 2x2 filter and stride 2
        H_conv_out = H // 2  # Division by 2 due to pooling with stride 2
        W_conv_out = W // 2  # Division by 2 due to pooling with stride 2

        # Flatten the input to each neurons
        conv_out = num_filters * H_conv_out * W_conv_out

        self.params['W2'] = np.random.normal(loc=0, scale=weight_scale, size=(conv_out, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        # print(X.shape)

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Forward pass for the three-layer conv net:
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        
        # First layer: conv - relu - 2x2 max pool
        # Using the conv_relu_pool_forward helper function
        conv_layer, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        # Second layer: affine - relu
        # We need to reshape the output from the conv layer to feed into the affine layer
        N, F, H_out, W_out = conv_layer.shape
        conv_layer_flat = conv_layer.reshape(N, F * H_out * W_out)
        hidden_layer, hidden_cache = affine_relu_forward(conv_layer_flat, W2, b2)
        
        # Third layer: affine (no activation, will be handled by softmax loss)
        scores, scores_cache = affine_forward(hidden_layer, W3, b3)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Compute the loss by softmax
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        # Backpropagation

        # Third layer: affine backward
        dhidden, dW3, db3 = affine_backward(dscores, scores_cache)

        # Add regularization
        dW3 += self.reg * W3

        # Second layer: affine_relu backward
        dconv_flat, dW2, db2 = affine_relu_backward(dhidden, hidden_cache)

        # Add regularization
        dW2 += self.reg * W2

        # Reshape gradient to match the conv layer output shape
        N, F, H_out, W_out = conv_layer.shape
        dconv = dconv_flat.reshape(N, F, H_out, W_out)

        # First layer: conv_relu_maxpool backward
        dx, dW1, db1 = conv_relu_pool_backward(dconv, conv_cache)
        
        dW1 += self.reg * W1

        grads = {
            'W1' : dW1, 'b1': db1,
            'W2' : dW2, 'b2': db2,
            'W3' : dW3, 'b3': db3
        }

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
