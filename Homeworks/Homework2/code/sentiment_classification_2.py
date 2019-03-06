import torch
from torchtext import data
import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)
word_to_ix = {}

def get_text_label(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    text = []
    label = []
    for line in lines:
        line_split = line[:-2].split()
        label.append(int(line_split[0]))
        text.append(line_split[1:])
    return text, label

def get_vocab(text):
    for t in text:
        for i in t:
            if i not in word_to_ix:
                word_to_ix[i] = len(word_to_ix)

def get_lookup(text, D):
    N = len(text)
    L = max_len(text)
    lookup_tensor = torch.zeros(N, L).long()
    mask = torch.zeros(N, L)
    for t in range(N):
        actual_size = len(text[t])
        lookup_tensor_t = torch.tensor([word_to_ix[i] for i in text[t]], dtype=torch.long)
        lookup_tensor[t,0:actual_size] = lookup_tensor_t
        mask[t, 0:actual_size] = 1 / actual_size

    return lookup_tensor, mask.view(N, L)

def max_len(text):
    max = 0
    for t in text:
        if len(t) > max:
            max = len(t)
    return max

def accuracy(out, labels):
    predictions = (out > 0.5)
    print("accuracy: ", torch.mean((predictions.view(len(labels)).float() == labels).float()))


text, label = get_text_label('data/train.txt')
get_vocab(text)
text_val, label_val = get_text_label('data/dev.txt')
text_test, label_test = get_text_label('data/test.txt')
labels = torch.FloatTensor(label)
labels_val = torch.FloatTensor(label_val)
labels_test = torch.FloatTensor(label_test)

N = len(text)
VOCAB_SIZE = len(word_to_ix)
N_list = list(range(N))
D = 50
lookup_tensor, mask = get_lookup(text, D)

class WEClassifier(nn.Module):

    def __init__(self, vocab_size, dimension):
        super(WEClassifier, self).__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, dimension)
        self.linear = nn.Linear(dimension, 1)

    def forward(self, lookup_tensor, mask):
        N, L = lookup_tensor.shape
        embeds = self.embeddings(lookup_tensor) * mask.view(N, L, 1)
        embeds_mean = torch.sum(embeds, 1) # mask already normalize
        return torch.sigmoid(self.linear(embeds_mean))

net = WEClassifier(VOCAB_SIZE, D)

net.zero_grad()

optimizer = optim.ASGD(net.parameters(), lr=0.01)
num_epochs = 100
batch_size = 10
batch_num = N // batch_size
for epoch in range(num_epochs):
    print("epoch ", epoch)
    rand_list = torch.randperm(N)
    for i in range(batch_num):
        optimizer.zero_grad() 
        batch_ids = rand_list[i * batch_size: (i+1) * batch_size]
        out = net.forward(lookup_tensor[batch_ids,:], mask[batch_ids,:])
        loss = F.mse_loss(out, labels[batch_ids])
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            out = net.forward(lookup_tensor, mask)
            print('train: ', F.mse_loss(out, labels))
            accuracy(out, labels)
            '''
            out_val = net.forward(lookup_tensor, mask)
            print('val: ', F.mse_loss(out_val, labels_val))
            accuracy(out_val, labels_val)
            '''

with torch.no_grad():
    out = net.forward(lookup_tensor, mask)
    print('test: ', F.mse_loss(out, labels_test))
    accuracy(out, labels_test)