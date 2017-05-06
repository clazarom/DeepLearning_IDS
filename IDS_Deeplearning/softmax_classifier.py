"""https://github.com/ohheydom/softmax_classifier"""

import numpy as np
import math
import random
from operator import itemgetter

class Softmax:
    def __init__(self, batch_size=50, epochs=1000, learning_rate=1e-2, reg_strength=1e-5, weight_update='adam'):
        self.W = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weight_update = weight_update

    def train(self, X, y):
        n_features = X.shape[1]
        n_classes = y.max() + 1
        self.W = np.random.randn(n_features, n_classes) / np.sqrt(n_features/2)
        config = {'reg_strength': self.reg_strength, 'batch_size': self.batch_size,
                'learning_rate': self.learning_rate, 'eps': 1e-8, 'decay_rate': 0.99,
                'momentum': 0.9, 'cache': None, 'beta_1': 0.9, 'beta_2':0.999,
                'velocity': np.zeros(self.W.shape)}
        c = globals()['Softmax']
        for epoch in range(self.epochs):
            loss, config = getattr(c, self.weight_update)(self, X, y, config)
            print ("Epoch:" +str(epoch)+", Loss: "+str(loss))

    def predict(self, X):
        return np.argmax(X.dot(self.W), 1)

    def loss(self, X, y, W, b, reg_strength):
        sample_size = X.shape[0]
        predictions = X.dot(W) + b

        # Fix numerical instability
        predictions -= predictions.max(axis=1).reshape([-1, 1])

        # Run predictions through softmax
        softmax = math.e**predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])

        # Cross entropy loss
        loss = -np.log(softmax[np.arange(len(softmax)), y]).sum() 
        loss /= sample_size
        loss += 0.5 * reg_strength * (W**2).sum()

        softmax[np.arange(len(softmax)), y] -= 1
        dW = (X.T.dot(softmax) / sample_size) + (reg_strength * W)
        return loss, dW

    def sgd(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength')(config)
        learning_rate, batch_size, reg_strength = items

        loss, dW = self.sample_and_calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)
 
        self.W -= learning_rate * dW
        return loss, config

    def sgd_with_momentum(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'momentum')(config)
        learning_rate, batch_size, reg_strength, momentum = items

        loss, dW = self.sample_and_calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        config['velocity'] = momentum*config['velocity'] - learning_rate*dW
        self.W += config['velocity']
        return loss, config

    def rms_prop(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'decay_rate', 'eps', 'cache')(config)
        learning_rate, batch_size, reg_strength, decay_rate, eps, cache = items

        loss, dW = self.sample_and_calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        cache = np.zeros(dW.shape) if cache == None else cache
        cache = decay_rate * cache + (1-decay_rate) * dW**2
        config['cache'] = cache

        self.W -= learning_rate * dW / (np.sqrt(cache) + eps)
        return loss, config

    def adam(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'eps', 'beta_1', 'beta_2')(config)
        learning_rate, batch_size, reg_strength, eps, beta_1, beta_2 = items
        config.setdefault('t', 0)
        config.setdefault('m', np.zeros(self.W.shape))
        config.setdefault('v', np.zeros(self.W.shape))

        loss, dW = self.sample_and_calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        config['t'] += 1
        config['m'] = config['m']*beta_1 + (1-beta_1)*dW
        config['v'] = config['v']*beta_2 + (1-beta_2)*dW**2
        m = config['m']/(1-beta_1**config['t'])
        v = config['v']/(1-beta_2**config['t'])
        self.W -= learning_rate*m/(np.sqrt(v)+eps)
        return loss, config

    def sample_and_calculate_gradient(self, X, y, batch_size, w, b, reg_strength):
        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        return self.loss(X_batch, y_batch, w, b, reg_strength)
