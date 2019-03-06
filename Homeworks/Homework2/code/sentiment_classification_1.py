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

unlabel_text = []
with open('data/unlabelled.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_split = line[:-2].split()
        unlabel_text.append(line_split)



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

def get_bow_vec_label(text, label):
    bow_vec = torch.zeros(len(text), len(word_to_ix))
    for t in range(len(text)):
        for i in text[t]:
            try:
                bow_vec[t, word_to_ix[i]] = 1
            except:
                pass
    labels = torch.FloatTensor(label)
    return bow_vec, labels

def accuracy(out, labels):
    predictions = (out > 0.5)
    print("accuracy: ", torch.mean((predictions.view(len(labels)).float() == labels).float()))

text, label = get_text_label('data/train.txt')
get_vocab(text)
bow_vec, labels = get_bow_vec_label(text, label)
text_val, label_val = get_text_label('data/dev.txt')
bow_vec_val, labels_val = get_bow_vec_label(text_val, label_val)
text_test, label_test = get_text_label('data/test.txt')
bow_vec_test, labels_test = get_bow_vec_label(text_test, label_test)

N = len(text)
VOCAB_SIZE = len(word_to_ix)


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, 1)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return torch.sigmoid(self.linear(bow_vec))


net = BoWClassifier(VOCAB_SIZE)
net.zero_grad()

optimizer = optim.ASGD(net.parameters(), lr=0.01)
num_epochs = 300
batch_size = 10
batch_num = N // batch_size
for epoch in range(num_epochs):
    print("epoch ", epoch)
    rand_list = torch.randperm(N)
    for i in range(batch_num):
        optimizer.zero_grad() 
        out = net.forward(bow_vec[rand_list[i * batch_size:(i+1) * batch_size],:])
        loss = F.mse_loss(out, labels[rand_list[i * batch_size:(i+1) * batch_size]])
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            out = net.forward(bow_vec)
            print('train: ', F.mse_loss(out, labels))
            accuracy(out, labels)
            out_val = net.forward(bow_vec_val)
            print('val: ', F.mse_loss(out_val, labels_val))
            accuracy(out_val, labels_val)

with torch.no_grad():
    out = net.forward(bow_vec_test)
    print('test: ', F.mse_loss(out, labels_test))
    accuracy(out, labels_test)


bow_vec_unlabel ,_ = get_bow_vec_label(unlabel_text, label)
out = net.forward(bow_vec_unlabel)
out = out > 0.5
with open('data/result.txt', 'w') as f:
    for i in out:
        f.write(str(i)[8] + "\n")