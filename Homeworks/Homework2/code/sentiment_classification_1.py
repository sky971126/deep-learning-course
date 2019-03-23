import torch
from torchtext import data
import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

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
    bow_vec = bow_vec
    labels = torch.FloatTensor(label)
    return bow_vec, labels

def accuracy(out, labels):
    predictions = (out > 0.5)
    return torch.mean((predictions.view(len(labels)).float() == labels).float())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
torch.tensor([1], device=device)


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


def train(net, epoch_num, batch_size, trainset, trainloader):
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(epoch_num):
        start = time.time()
        print("epoch ", epoch)
        for data in trainloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad() 
            out = net.forward(x).view(-1)
            loss = F.binary_cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == epoch_num - 1:
            with torch.no_grad():
                total_accuracy = 0
                total_accuracy_val = 0
                for data in trainloader:
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    out = net.forward(x).view(-1)
                    total_accuracy += accuracy(out, y)
                print('train: ', total_accuracy / len(trainloader))
        end = time.time()
        print('eplased time', end-start)

def test(net, batch_size, testset, testloader):
    with torch.no_grad():
        total_accuracy = 0
        for data in testloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            out = net.forward(x)
            total_accuracy += accuracy(out, y)
        print('test: ', total_accuracy / len(testloader))

net = BoWClassifier(VOCAB_SIZE).to(device)
net.zero_grad()

epoch_num = 50
batch_size = 1000
trainset = torch.utils.data.TensorDataset(bow_vec, labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torch.utils.data.TensorDataset(bow_vec_test, labels_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

train(net, epoch_num, batch_size, trainset, trainloader)
test(net, batch_size, testset, testloader)

bow_vec_unlabel ,_ = get_bow_vec_label(unlabel_text, label)
bow_vec_unlabel = bow_vec_unlabel.to(device)
out = net.forward(bow_vec_unlabel)
out = out > 0.5
with open('data/result_1.txt', 'w') as f:
    for i in out:
        f.write(str(i)[8] + "\n")