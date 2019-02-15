from mnist import MNIST
import numpy as np
import pickle
from optim import *
from layers import *
from solver import *
from svm import *
from cnn import *

mndata = MNIST('mnist')

images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)
X = images.reshape(60000,1,28,28)
y = labels
N = 54000
data = {
    'X_train': X[:N,:,:,:],
    'y_train': y[:N],
    'X_val': X[N:int(1.1*N),:,:,:],
    'y_val': y[N:int(1.1*N)]
}


model = ConvNet(num_filters=3, hidden_dim=16, filter_size=7, reg=0)#kernel 1
solver = Solver(model, data,
                update_rule='adam',
                optim_config={
                'learning_rate': 1e-3
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=10)
solver.train()
