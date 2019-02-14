import numpy as np
import pickle
from optim import *
from layers import *
from solver import *
from svm import *
from logistic import *


with open('data.pkl', 'rb') as f:
    data_all = pickle.load(f, encoding='latin1')
data = {
    'X_train': data_all[0][:500],
    'y_train': data_all[1][:500],
    'X_val': data_all[0][500:750],
    'y_val': data_all[1][500:750],
    'X_test': data_all[0][750:1000],
    'y_test': data_all[1][750:1000]
}


model = LogisticClassifier(input_dim=data_all[0].shape[1], hidden_dim=32, reg=0.3)
solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={
                'learning_rate': 8e-2,
                },
                lr_decay=0.9,
                num_epochs=20, batch_size=50,
                print_every=100)
solver.train()

"""
x = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
y = np.array([[1,0],[2,1],[3,0]])
z = np.array([[1,2,3],[8,9,10]])

print(x)
print(y)
print(x.dot(y))

print(np.mean(x,0))
r = np.random.random_sample((5, 2))
print(x.shape)
print(np.transpose(x))
N=10
C=3
F=6
s=7
a = np.ones((N,C,F))
b = np.ones((C,F,s))
print(np.dot(a,b).shape)
"""