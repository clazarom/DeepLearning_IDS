from sklearn.neural_network import MLPClassifier
# MORE INFO: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# Training data: X = n_samples x m_features
# NOTE
X = [[0., 0.],
     [1., 1.],
     [2., 2.],
     [-5., -1]]
# Target values, class values - 2 classes output y = 1 x n_samples
y = [0, 1, 1, 0]

#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)

#hidden_layer =  1x (# hidden layers - 2) . Each value: units in the hidden layer
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 5, 2), random_state=1, max_iter = 1500, verbose = True)
#Fit / Training with either algorithm as solver:
#   -  Stochastic Gradient Descent
#   -  Adam:  refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
#           -->  works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. 
#   -  L-BFGS: optimizer in the family of quasi-Newton methods.
#           --> For small datasets, however, ‘lbfgs’ can converge faster and perform better.
clf.fit(X, y)

prediction = clf.predict([[2., 2.], [-1., -2.], [10., 10.]])
print (prediction)

