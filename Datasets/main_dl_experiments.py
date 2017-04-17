import sys
import numpy as np
from MLP_nets import MLP_1x16, separateClasses
import SYS_VARS
import load_dataset as kdd
from autoencoders import SparseAutoencoder
import scipy.optimize


###########################################################################################
""" Normalize the dataset provided as input """

def normalizeDataset(dataset):

    """ Remove mean of dataset """
    dataset = dataset - np.mean(dataset)
    
    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """
    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev
    
    """ Rescale from [-1, 1] to [0.1, 0.9] """
    dataset = (dataset + 1) * 0.4 + 0.1
    
    return dataset

###########################################################################################
""" Randomly samples image patches, normalizes them and returns as dataset """

def loadDataset(num_patches, patch_side):

    """ Load KDD """
    file_10 = SYS_VARS.KDDCup_path_train_10
    
    """ Initialize dataset as array of zeros """
    dataset = np.zeros((patch_side*patch_side, num_patches))
    
    """ Normalize and return the dataset """
    dataset = normalizeDataset(dataset)
    return dataset



###########################################################################################
""" Loads data, trains the Autoencoder and visualizes the learned weights """

def executeSparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, training_data):  
    
    
    """ Initialize the Autoencoder with the above parameters """
    encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)
    opt_solution = encoder.train (training_data, max_iterations)
    #opt_solution  = scipy.optimize.minimize(encoder, encoder.theta, args = (training_data,), method = 'L-BFGS-B', jac = True, options = {'maxiter': max_iterations, 'disp' : True})

    if (opt_solution.success):
        print (opt_solution.message)
    # Recover parameters from theta = (W1_grad.flatten(), W2_grad.flatten(), b1_grad.flatten(), b2_grad.flatten())
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    opt_W2        = opt_theta[encoder.limit1 : encoder.limit2].reshape(visible_size, hidden_size)
    opt_b1        = opt_theta[encoder.limit2 : encoder.limit3].reshape(hidden_size, 1)
    opt_b2        = opt_theta[encoder.limit3 : encoder.limit4].reshape(visible_size, 1)

    """ Compute one data sample: input vs output """
    print("\nInput value ")
    x_in = training_data[:,4:5]
    #x_in= training_data.take([[:,5],[5]])
    print (x_in)
    opt_H, opt_X = encoder.computeOutputLayer(opt_W1, opt_W2, opt_b1, opt_b2, x_in)
    
    """visualizeW1(opt_W1, vis_patch_side, hid_patch_side)"""
    print("\nOutput value " )
    print (opt_X)

def executeMLP(train_data, y):
     # 1 HIDDEN LAYER, 64 
    mlp_16 = MLP_1x16()    
    print ("Training..."+str(y.shape) +  " and " +str(train_data.shape))
    mlp_16.train(np.transpose(train_data), y, 'trial')
    print ("To test..."+str(train_data[:, 4:5].shape[0])+"x"+str(train_data[:, 4:5].shape[1]))
    prediction16 = mlp_16.test(np.transpose(train_data[:, 4:5]))
    print ("Prediction: " + str(prediction16))
    for value in prediction16:
        for k in kdd.attacks_map.keys():
            if (int(value) == kdd.attacks_map[k]):
                print(k)

###########################################################################
def main():
    """ Load KDD dataset"""
    pre_data = np.transpose(kdd.simple_preprocessing_KDD())
    training_data = normalizeDataset(pre_data)

    ######## SPARSE AUTOENCODER TRAINING
    """ Define the parameters of the Autoencoder """
    """ rho            = 0.01   # desired average activation of hidden units... should be tuned!!
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 400    # number of optimization iterations
    visible_size = 42       # input & output sizes: KDDCup99 # of features
    hidden_size = 50        # sparse-autoencoder benefits from larger hidden layer units """
    """ Train and test a sample of the system """
    """executeSparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, training_data)
    print ("predata")
    for val in pre_data[:, 4:5]:
        print (str(val)) """
        
    ######## MULTILAYER PERCEPTRONS
    y =  separateClasses(pre_data, kdd._ATTACK_INDEX_KDD)
    executeMLP(pre_data, y)


if __name__ == "__main__":main() ## with if
