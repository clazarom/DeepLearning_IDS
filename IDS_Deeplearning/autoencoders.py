# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2013 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com
# Github repo: https://github.com/siddharth-agrawal/Sparse-Autoencoder

import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot


###########################################################################################
""" The Sparse Autoencoder class """

class SparseAutoencoder(object):

    #######################################################################################
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
    
        """ Initialize parameters of the Autoencoder object """
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.rho = rho                      # desired average activation of hidden units
        self.lamda = lamda                  # weight decay parameter
        self.beta = beta                    # weight of sparsity penalty term
        

        """ Set limits for accessing 'theta' values """
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size # W1 size: hidden x visible
        self.limit2 = 2 * hidden_size * visible_size # W2 size: hidden x visible
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size


        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Initialize optimal parameters """
        self.opt_W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        self.opt_W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        self.opt_b1 = numpy.zeros((hidden_size, 1))
        self.opt_b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """
        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    #######################################################################################
    """ Returns elementwise sigmoid output of input array """
    
    def sigmoid(self, x):
    
        return (1 / (1 + numpy.exp(-x)))

    #######################################################################################
    """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """
        
    def sparseAutoencoderCost(self, theta, input):
        """ Extract weights and biases from 'theta' input """
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        hidden_layer, output_layer = self.compute_layer(input, W1, W2, b1, b2)
        #hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        #output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)
        
        """ Estimate the average activation value of the hidden layers """
        
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        
        diff = output_layer - input
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))
        """ Sparse representation of inputs: Comparing the probability distribution of the hidden unit activations with some low desired value"""
        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                    (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))#Kullback–Leibler divergence
        cost                 = sum_of_squares_error + weight_decay + KL_divergence


        #GRADIENTS
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        
        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
            
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        
        """ Transform numpy matrices into arrays """
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
        print ("- Cost: "+ str(cost))                           
        return [cost, theta_grad]

    ###########################################################################################
    """ Set deault net parameters """
    def set_inner_weightsBiases(self, W1, W2, b1, b2):
        self.opt_W1 = W1
        self.opt_W2 = W2
        self.opt_b1 = b1
        self.opt_b2 = b2
    """Compute one sample of the dataset - input is a 2d array, with 1 column
        [[value1], [value2], ... [valueN]]
        row = datast[:, 1:2]"""
    def compute_layer(self, input, W1, W2, b1, b2):
        hidden_layer = self.sigmoid(numpy.add(numpy.dot(W1, input),b1))
        output_layer = self.sigmoid(numpy.add(numpy.dot(W2, hidden_layer),b2))
        #print ("Input: "+str(input.shape[0]) +" x "+str(input.shape[1]))
        #print ("Hidden: "+str(hidden_layer.shape[0]) +" x "+str(hidden_layer.shape[1]))
        #print ("Output: "+str(output_layer.shape[0])+" x " +str(output_layer.shape[1]))
        return hidden_layer, output_layer

    def do_nothing(self, input):
        return input

    """Compute one sample of the dataset - input is a 1d array vector
        [value1, value2, ... valueN]
        row = dataset[:, 2]"""    
    def compute_function(self, input):
        hidden_layer = self.sigmoid(numpy.add(numpy.dot(self.opt_W1, input.reshape(input.shape[0], -1)),self.opt_b1))
        output_layer = self.sigmoid(numpy.add(numpy.dot(self.opt_W2, hidden_layer),self.opt_b2))
        #return output_layer.flatten()
        return [x for x in output_layer]
        #return reshape(output_layer.shape[0], -1)

    
    def compute_dataset(self, input, W1, W2, b1, b2):
        """y = []
        for index in range(input.shape[1]):
            y.append(self.compute_layer(input[:,index], W1, W2, b1, b2))
        y_mat= numpy.transpose(numpy.array(y))
        return y_mat"""
        self.set_inner_weightsBiases(W1, W2, b1, b2)
        output = []
        output = numpy.apply_along_axis(self.compute_function, axis=0, arr=input) #apply compute_function for each sample
        #for index in range(input.shape[1]):
            #output.append(self.compute_layer(input[:,index], W1, W2, b1, b2))
        #output= numpy.transpose(numpy.array(output))

        return output

    ############################################################################################
    def train(self, training_data, max_iterations, algorithm = 'L-BFGS-B'):
         """ Run the L-BFGS algorithm to get the optimal parameter values """
         print("\n OPTIMIZATION " +str(training_data.shape[0]) +' x '+str(training_data.shape[1]))
         opt_solution  = scipy.optimize.minimize(self.sparseAutoencoderCost, self.theta, args = (training_data,), method = algorithm, jac = True, options = {'maxiter': max_iterations, 'disp' : True})
         return opt_solution


