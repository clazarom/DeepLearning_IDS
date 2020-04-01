import sys
from MLP_nets import MLP_1x64, MLP_1x16, MLP_2

def main():
    """MOCK dataset: 2 features, 2 outputs (0, 1)"""
    X = [[0., 0.],
         [1., 1.],
         [2., 2.],
         [-5., -1]]
    """Target values, class values - 2 classes output y = 1 x n_samples"""
    y = [0, 1, 1, 0]
    X_test = [[2., 2.], [-1., -2.], [10., 10.]]
    
    # 1 HIDDEN LAYER, 64 
    mlp_64 = MLP_1x64()    
    print ("Training...")
    mlp_64.train(X, y, 'trial')
    prediction64 = mlp_64.test(X_test)

    # 1 HIDDEN LAYER, 16 
    mlp_16 = MLP_1x16()
    print ("Training...")
    mlp_16.train(X, y, 'trial')
    prediction16 = mlp_16.test(X_test)

    # 2 HIDDEN LAYER {41 units, 5 units}
    mlp_2 = MLP_2()
    print ("Training...")
    mlp_2.train(X, y, 'trial')
    prediction2 = mlp_2.test(X_test)
    

    """Print results"""
    print ("Prediction:")
    print ("1x64: "+ str(prediction64)
           +", 1x16: "+ str(prediction16)
           +", 2x{41, 5}: "+ str(prediction2))

if __name__ == "__main__":main() ## with if
