import sys
import numpy as np
import SYS_VARS
import load_dataset as kdd
from autoencoders import SparseAutoencoder
from MLP_nets import MLP_general
import scipy.optimize
import analysis_functions


###########################################################################################
""" Normalize the dataset provided as input """

def normalize_dataset(dataset):

    """ Remove mean of dataset """
    dataset = dataset - np.mean(dataset)
    
    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """
    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev
    
    """ Rescale from [-1, 1] to [0.1, 0.9] """
    dataset = (dataset + 1) * 0.4 + 0.1
    
    return dataset


###########################################################################################
""" Loads data, trains the Autoencoder and visualizes the learned weights """

def execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, train_data):  
    
    
    """ Initialize the Autoencoder with its parameters and train"""
    encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)
    #opt_solution  = scipy.optimize.minimize(encoder, encoder.theta, args = (training_data,), method = 'L-BFGS-B', jac = True, options = {'maxiter': max_iterations, 'disp' : True})
    opt_solution = encoder.train (train_data, max_iterations)
    if (opt_solution.success):
        print (opt_solution.message)
        
    # Recover parameters from theta = (W1_grad.flatten(), W2_grad.flatten(), b1_grad.flatten(), b2_grad.flatten())
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    opt_W2        = opt_theta[encoder.limit1 : encoder.limit2].reshape(visible_size, hidden_size)
    opt_b1        = opt_theta[encoder.limit2 : encoder.limit3].reshape(hidden_size, 1)
    opt_b2        = opt_theta[encoder.limit3 : encoder.limit4].reshape(visible_size, 1)

    # Compute one data sample: input vs output
    """print("\nInput value ")
    x_in = training_data[:,4:5]
    #x_in= training_data.take([[:,5],[5]])
    print (x_in)
    opt_H, opt_X = encoder.compute_layer(x_in, opt_W1, opt_W2, opt_b1, opt_b2)"""
    
    """visualizeW1(opt_W1, vis_patch_side, hid_patch_side)"""
    """print("\nOutput value " )
    print (opt_X)"""

    # Return input dataset computed with autoencoder
    return encoder.compute_dataset(train_data, opt_W1, opt_W2, opt_b1, opt_b2)

def execute_MLP(train_data, y, classes_names):
     # 1 HIDDEN LAYER, 16 
    mlp = MLP_general(16)    
    print ("Training..."+str(y.shape) +  "(labels) and " +str(train_data.shape)+"(dataset)")
    mlp.train(np.transpose(train_data), y, 'trial')
    print ("To test..."+str(train_data[kdd._ATTACK_INDEX_KDD, 4:5]))
    # Compute one sample - train_data[:, 4:5]
    """prediction16 = mlp_16.test(np.transpose(train_data[:, 4:5]))
    print ("Prediction: " + str(prediction16))
    for value in prediction16:
        for k in kdd.attacks_map.keys():
            if (int(value) == kdd.attacks_map[k]):
                print(k)"""
    y_data = mlp.compute_dataset(train_data)
    # Validate
    analysis_functions.validation(mlp.classifier, train_data, y_data, y, classes_names)
    #analysis_functions.print_totals(y_data, y)

    return y_data

###########################################################################

    
def main():
    
    ####### LOAD KDD dataset 
    pre_data = np.transpose(kdd.simple_preprocessing_KDD())
    x_train, y, classes_names =  kdd.separate_classes(pre_data, kdd._ATTACK_INDEX_KDD)
    x_train_normal = normalize_dataset(x_train)

    print("Data preprocessing results:" )
    print("indata "+str(pre_data.shape[1]))
    print("classfied "+str(x_train.shape[1]))
    print("normalized " +str(x_train_normal.shape[1]))


    """move = []
    for index in range(training_data.shape[1]):
        #np.append(move,training_data[:,index], axis = 0)
        move.append(training_data[:,index])
    move_mat= np.transpose(np.array(move))
    print (str(move_mat.shape[1]))"""

    ######## SPARSE AUTOENCODER TRAINING
    """ Define the parameters of the Autoencoder """
    rho            = 0.01   # desired average activation of hidden units... should be tuned!!
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 400    # number of optimization iterations
    visible_size = 41       # input & output sizes: KDDCup99 # of features
    hidden_size = 50        # sparse-autoencoder benefits from larger hidden layer units """
    """ Train and test a sample of the system """
    featured_x = execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, x_train_normal)
    
        
    ######## MULTILAYER PERCEPTRONS
    y_predicted = execute_MLP(featured_x, y, classes_names)


    print(str(y_predicted[1]) +" vs real: "+ str(y[1]))


if __name__ == "__main__":main() ## with if
