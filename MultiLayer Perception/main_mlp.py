import sys
from MLP_nets import MLP_1x64, MLP_1x16

def main():
    #MOCK dataset: 2 features, 2 outputs (0, 1)
    X = [[0., 0.],
         [1., 1.],
         [2., 2.],
         [-5., -1]]
    # Target values, class values - 2 classes output y = 1 x n_samples
    y = [0, 1, 1, 0]
    X_test = [[2., 2.], [-1., -2.], [10., 10.]]
    
    # HIDDEN LAYER, 64 
    mlp_64 = MLP_1x64()    
    print ("Training...")
    mlp_64.train(X, y, 'trial')
    prediction64 = mlp_64.test(X_test)

    # HIDDEN LAYER, 16 
    mlp_16 = MLP_1x16()
    print ("Training...")
    mlp_16.train(X, y, 'trial')
    prediction = mlp_16.test(X_test)

    #Print results
    print ("Prediction:")
    print ("64: "+ str(prediction64) +", 16: "+ str(prediction))

if __name__ == "__main__":main() ## with if
