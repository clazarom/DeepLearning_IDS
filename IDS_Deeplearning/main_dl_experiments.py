import sys
import numpy as np
from sklearn import preprocessing
import SYS_VARS
import load_dataset as kdd
from autoencoders import SparseAutoencoder
from MLP_nets import MLP_general
import scipy.optimize
import analysis_functions
from softmax_classifier import Softmax


###########################################################################################

def normalize_dataset(dataset):
    """ Normalize the dataset provided as input, values in scalte 0 to 1 """
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(dataset)

def sparse_normalize_dataset(dataset):
    """ Normaliza dataset without removing the sparseness structure of the data """
    #Remove mean of dataset 
    dataset = dataset - np.mean(dataset)
    #Truncate to +/-3 standard deviations and scale to -1 to 1
    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev
    #Rescale from [-1, 1] to [0.1, 0.9]
    dataset = (dataset + 1) * 0.4 + 0.1
    #dataset = (dataset-np.amin(dataset))/(np.amax(dataset)-np.amin(dataset))
    return dataset
    #return preprocessing.MaxAbsScaler().fit_transform(dataset)



def softmax (x):
    """ Compute softmax values for each sets of scores in x.
        S(xi) =  exp[xi] / Sum {exp[xi]} 
    """
    assert len(x.shape) == 2
    s = np.max(x, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(x - s)
    s_denom = np.sum(e_x, axis=1)
    s_denom = s_denom[:, np.newaxis] # dito
    return e_x / s_denom



###########################################################################################

def execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, train_data, test_data):  
    """ Trains the Autoencoder with the trained data and parameters and returns train_data after the network """
    
    #Initialize the Autoencoder with its parameters and train
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
    train =  encoder.compute_dataset(train_data, opt_W1, opt_W2, opt_b1, opt_b2)
    test =  encoder.compute_dataset(test_data, opt_W1, opt_W2, opt_b1, opt_b2)
    sample = []#sample = encoder.compute_layer(test_data[:, 0:1], opt_W1, opt_W2, opt_b1, opt_b2)
    
    return train, test, sample


def execute_MLP(train_data, hidden_layers, y, test_data):
    """ Trains the MLP with the train_data and returns train_data after the network """
    # Train
    mlp = MLP_general(hidden_layers)    
    #print ("Training..."+str(y.shape) +  "(labels) and " +str(train_data.shape)+"(dataset)")
    mlp.train(np.transpose(train_data), y, 'trial')
    #print ("To test..."+str(train_data[kdd._ATTACK_INDEX_KDD, 4:5]))
    # Compute for all train_data
    """ Compute one sample - train_data[:, 4:5]
    prediction1 = mlp.test(np.transpose(train_data[:, 4:5]))
    print ("Prediction: " + str(prediction16))
    for value in prediction1:
        for k in kdd.attacks_map.keys():
            if (int(value) == kdd.attacks_map[k]):
                print(k)"""
    y_train = mlp.compute_dataset(train_data)
    y_test = mlp.compute_dataset(test_data)
    #analysis_functions.print_totals(y_data, y)

    return mlp, y_train, y_test


########### IDS with DEEPLEARNING #############################

def ids_mlp(train_data, y_in, test_data):
        
    ######## MULTILAYER PERCEPTRONS
    h_layers = [64]          # hidden layers, defined by their sizes {i.e 2 layers with 30 and 20 sizes [30, 20]}
    mlp, y_predicted, y_predicted_test = execute_MLP(train_data, hidden_layers = h_layers, y =  y_in, test_data=test_data)
    print(str(y_predicted[1]) +" vs real: "+ str(y_in[1]))

    return mlp, y_predicted, y_predicted_test

def deeplearning_sae_mlp(train_data, y_train, test_data, features):

    ######## SPARSE AUTOENCODER TRAINING
    """ Define the parameters of the Autoencoder """
    rho            = 0.01   # desired average activation of hidden units... should be tuned!!
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 400    # number of optimization iterations
    visible_size = features       # input & output sizes: KDDCup99 # of features
    hidden_size = 50        # sparse-autoencoder benefits from larger hidden layer units """
    """ Train and do a test over the same traindata """
    featured_x, f_test_x, f_sample = execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, train_data, test_data)
    
        
    ######## MULTILAYER PERCEPTRONS
    h_layers = [64]          # hidden layers, defined by their sizes {i.e 2 layers with 30 and 20 sizes [30, 20]}
    mlp, y_predicted, y_predicted_test = execute_MLP(featured_x, hidden_layers = h_layers, y = y_train, test_data = f_test_x)
    y_sample = []#y_sample = mlp.test(np.transpose(f_sample.reshape(f_sample.shape[0], -1)))


    #Sprint(str(y_predicted[1]) +" vs real: "+ str(y_train[1])+" vs one sample:"+str(y_sample))
    return mlp, y_predicted, y_predicted_test

def deeplearning_sae_sae(train_data, y_train, test_data, features):

    ######## SPARSE AUTOENCODER TRAINING
    """ Define the parameters of the Autoencoder """
    rho            = 0.01   # desired average activation of hidden units... should be tuned!!
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 400    # number of optimization iterations
    visible_size = features       # input & output sizes: KDDCup99 # of features
    hidden_size = 50        # sparse-autoencoder benefits from larger hidden layer units """
    """ Train and do a test over the same traindata """
    featured_x, f_test_x, f_sample = execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, train_data, test_data)
    
        
    ######## SPARSE AUTOENCODER AND SOFTMAX FOR Classification
    """ Define the parameters of the Autoencoder """
    rho            = 0.01   # desired average activation of hidden units... should be tuned!!
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 400    # number of optimization iterations
    visible_size = features      # input & output sizes: KDDCup99 # of features
    hidden_size = 16        # sparse-autoencoder benefits from larger hidden layer units """

    """ Train and do a test over the same traindata """
    y_prima_train, y_prima_test, f_sample = execute_sparseAutoencoder(rho, lamda, beta, max_iterations, visible_size, hidden_size, featured_x, f_test_x)

    """Softmax classifier """
    #y_predicted = softmax(y_prima_train)
    #y_predicted_test = softmax(y_prima_test)
    reg_strength = 1e-4
    batch_size = train_data.shape[0]
    epochs = 1000
    learning_rate = 1
    weight_update = 'sgd_with_momentum'
    clf = Softmax(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, reg_strength=reg_strength, weight_update=weight_update)
    clf.train(np.transpose(train_data), y_train)
    y_predicted = clf.predict(np.transpose(train_data))
    y_predicted_test = clf.predict(np.transpose(test_data))
    print (str(np.mean(np.equal(y_train, y_predicted))))
    return y_predicted, y_predicted_test, clf


###########################################################################

    
def main():
    """test soft max -solution
    without normalization: [[ 0.00626879  0.01704033  0.04632042  0.93037047]
                           [ 0.01203764  0.08894682  0.24178252  0.65723302]
                           [ 0.00626879  0.01704033  0.04632042  0.93037047]]
    x2 = np.array([[1, 2, 3, 6],  # sample 1
               [2, 4, 5, 6],  # sample 2
               [1, 2, 3, 6]]) # sample 1 again(!)
    print(softmax(sparse_normalize_dataset(x2)))"""
    # LOAD KDD dataset & preprocess
    select_dataset = kdd._ATTACK_INDEX_KDD # kdd._ATTACK_INDEX_NSLKDD
    dataset_features = 41 #42
    pre_train_data, pre_test_data = kdd.simple_preprocessing_KDD(select_dataset)
    pre_train_data = np.transpose(pre_train_data)
    pre_test_data = np.transpose(pre_test_data)
    
    x_train, y, classes_names, classes_values =  kdd.separate_classes(pre_train_data, select_dataset)
    y = np.array([int(i) for i in y]) #convert floats to integer

    if (select_dataset == kdd._ATTACK_INDEX_NSLKDD):
        x_test, y_test, classes_names, classes_values =  kdd.separate_classes(pre_test_data, select_dataset)
        y_test = np.array([int(i) for i in y_test]) #convert floats to integer

    else:
        x_test = pre_test_data
        y_test = y

    x_train_normal_s = sparse_normalize_dataset(x_train)
    x_train_normal = normalize_dataset(x_train)
    
    x_test_normal_s = sparse_normalize_dataset(x_test)
    x_test_normal = normalize_dataset(x_test)

    #Data load and preprocessing plot results
    print("Data preprocessing results(train):" )
    print("indata "+str(pre_train_data.shape)
          +", classfied "+str(x_train.shape)
          +"normalized " +str(x_train_normal_s.shape))
    print("Data preprocessing results(test):" )
    print("indata "+str(pre_train_data.shape)
          +", classfied "+str(x_train.shape)
          +"normalized " +str(x_train_normal_s.shape))
    print ("Output classes: ")
    for c in classes_names:
        print (c)
    #Debug - plot all attacks kdd.plot_various()
    #kdd.plot_percentages(classes_values, classes_names, 'Train', y)


    """move = []
    for index in range(training_data.shape[1]):
        #np.append(move,training_data[:,index], axis = 0)
        move.append(training_data[:,index])
    move_mat= np.transpose(np.array(move))
    print (str(move_mat.shape[1]))"""
    
    #First deep learning architecture: SAE (feature selection) and MLP (classifier)
    mlp, y_n1, y_n1_test = deeplearning_sae_mlp(x_train_normal_s, y, x_test_normal_s, dataset_features)
    #No feature selection: Only MLP
    mlp_solo, y_standalone1, y_standalone1_test = ids_mlp(x_train_normal, y, x_test_normal)

    #Second deep learning architecture: SAE (feature selection) and SAE-softmax (classifier)
    y_n2, y_n2_test, sm_classifier = deeplearning_sae_sae(x_train_normal_s, y, x_test_normal, dataset_features)

    #Validation
    print("\nSAE and MLP Validation "+str(y_n1.shape))
    analysis_functions.validation(mlp.classifier, x_train_normal_s, y_n1, y, classes_names, ' KDD SAE-MLP(train)')
    #NSL-KDD only analysis_functions.validation(mlp.classifier, x_test_normal_s, y_n1_test, y_test, classes_names, ' KDD SAE-MLP(test)')
    print("\nMLP only Validation "+str(y_standalone1.shape))
    analysis_functions.validation(mlp_solo.classifier, x_train_normal, y_standalone1, y, classes_names, ' KDD MLP(train)')
    #NSL-KDD only analysis_functions.validation(mlp_solo.classifier, x_test_normal, y_standalone1_test, y_test, classes_names, ' KDD MLP(test)')

    print("\nSAE and SAE-softmax Validation" +str(y_n2.shape))
    analysis_functions.validation(sm_classifier, x_train_normal_s, y_n2, y, classes_names, ' KDD SAE-SAE(train)')
    #NSL-KDD only analysis_functions.validation(sm_classifier, x_test_normal_s, y_n2_test, y_test, classes_names, ' NSLKDD SAE-SAE(test)')


   


if __name__ == "__main__":main() ## with if
