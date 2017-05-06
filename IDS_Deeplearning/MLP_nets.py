from sklearn.neural_network import MLPClassifier
from sklearn import model_selection # cross_val_score
from sklearn.metrics import precision_score, precision_recall_fscore_support
import numpy as np
# MORE INFO: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# Training data: X = n_samples x m_features
# NOTE


""" Generic Multilayer Perceptron"""
class MLP_general(object):

    """ Create and set the layers with their sizes i.e. (MLP_general(10, 20) has 2 layers of sizes 10 and 20)"""
    def __init__(self, *layers, a =1e-5, max_i = 1500):
        self.classifier = MLPClassifier(solver='adam', alpha=a, *layers, random_state=1, max_iter = max_i, verbose = True)

    def train(self, X, y, dataset):    
        self.classifier.fit(X, y)

    def test (self, X_test):
        return self.classifier.predict(X_test)

    def test_batch (self, X_test):
        return self.classifier.predict(np.transpose(X_test.reshape(X_test.shape[0], -1))).flatten()

    def do_nothing(self, X_test):
        return X_test

    def compute_dataset(self, input):
        return np.apply_along_axis(self.test_batch, axis=0, arr=input).flatten()

    def validation(self, data, y_data, y_target):
        #kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        #cv = kfold
        x =  np.transpose(data)
        accuracy = model_selection.cross_val_score(self.classifier, x, y_target, scoring='accuracy')
        #precision = model_selection.cross_val_score(self.classifier, x, target, scoring='precision')
        #precision_score(y_true, y_pred, average='macro')  
        #recall = model_selection.cross_val_score(self.classifier, x, target, scoring='recall')
        precision, recall, fscore, m = precision_recall_fscore_support(y_target, y_data, average='macro')
        print("MLP Validation:")
        print(str(accuracy[0]) +", " +str(precision) +", " +str(recall))


########################################################################
    

    
            
       
