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
N = 54000
data = {
    'X_train': X[:N,:],
    'y_train': y[:N],
    'X_val': X[N:int(1.1*N),:],
    'y_val': y[N:int(1.1*N)]
}


model = SoftmaxClassifier(hidden_dim=64, reg=0.02)
solver = Solver(model, data,
                update_rule='rmsprop',
                optim_config={
                'learning_rate': 1e-4,
                },
                lr_decay=0.9,
                num_epochs=50, batch_size=100,
                print_every=1000)
solver.train()