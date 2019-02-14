import numpy as np
import pickle
from optim import *
from layers import *
from solver import *
from svm import *
from logistic import *
from softmax import *
from mnist import MNIST

mndata = MNIST('mnist')

images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)
X = images.reshape(60000,28*28)
y = labels
N = 50000
data = {
    'X_train': X[:N,:],
    'y_train': y[:N],
    'X_val': X[N:int(1.1*N),:],
    'y_val': y[N:int(1.1*N)]
}


model = SoftmaxClassifier(hidden_dim=None, reg=0.01)
solver = Solver(model, data,
                update_rule='adam',
                optim_config={
                'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=50,
                print_every=1000)
solver.train()