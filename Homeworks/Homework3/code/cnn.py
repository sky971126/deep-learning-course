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

def get_lookup(text, max_len, D):
    N = len(text)
    L = max_len
    lookup_tensor = torch.zeros(N, L).long()
    mask = torch.zeros(N, L)
    for t in range(N):
        actual_size = min(len(text[t]), L)
        for i in range(actual_size):
            try:
                lookup_tensor[t,i] = word_to_ix[text[t][i]]
            except:
                pass
        mask[t, 0:actual_size] = 1

    return lookup_tensor, mask.view(N, L)

def max_len(text):
    max = 0
    for t in text:
        if len(t) > max:
            max = len(t)
    return max

def accuracy(out, labels):
    predictions = (out > 0.5)
    return torch.mean((predictions.view(len(labels)).float() == labels).float())

def train(net, epoch_num, batch_size, trainset, trainloader):
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=0.0005)
    for epoch in range(epoch_num):
        start = time.time()
        print("epoch ", epoch)
        for data in trainloader:
            x, y, mask = data
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            out = net.forward(x, mask).view(-1)
            loss = F.binary_cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == epoch_num - 1:
            with torch.no_grad():
                total_accuracy = 0
                total_accuracy_val = 0
                for data in trainloader:
                    x, y, mask = data
                    x = x.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                    out = net.forward(x, mask)
                    total_accuracy += accuracy(out, y)
                print('train: ', round(100 * total_accuracy.item() / len(trainloader), 2), '%')
        end = time.time()
        print('eplased time', end-start)

def test(net, batch_size, testset, testloader):
    with torch.no_grad():
        total_accuracy = 0
        for data in testloader:
            x, y, mask = data
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            out = net.forward(x, mask)
            total_accuracy += accuracy(out, y)
        print('test: ', round(100 * total_accuracy.item() / len(testloader), 2), '%')

class CNN(nn.Module):

    def __init__(self, vocab_size, in_channel, out_channel, kernel_size, average=True):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, in_channel)
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size)
        self.linear = nn.Linear(out_channel, 1)
        self.average = average

    def forward(self, lookup_tensor, mask):
        N, L = lookup_tensor.shape
        embeds = self.embeddings(lookup_tensor) * mask.view(N, L, 1)
        cnn_out = self.conv(embeds.transpose(2,1))
        if self.average:
            pool_out = torch.mean(cnn_out, 2)
        else:
            pool_out = torch.max(cnn_out, 2)[0]
        relu_out = F.relu(pool_out)
        return torch.sigmoid(self.linear(relu_out))

text, label = get_text_label('data/train.txt')
get_vocab(text)
text_val, label_val = get_text_label('data/dev.txt')
text_test, label_test = get_text_label('data/test.txt')
labels = torch.FloatTensor(label)
labels_val = torch.FloatTensor(label_val)
labels_test = torch.FloatTensor(label_test)

L = max_len(text)
VOCAB_SIZE = len(word_to_ix)
in_channel = 300
out_channel = 256
kernel_size = 7
lookup_tensor, mask = get_lookup(text, L, in_channel)
lookup_tensor_test, mask_test = get_lookup(text_test, L, in_channel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
torch.tensor([1], device=device)

net = CNN(VOCAB_SIZE, in_channel, out_channel, kernel_size, average=True).to(device)
net.zero_grad()

epoch_num = 50
batch_size = 500

trainset = torch.utils.data.TensorDataset(lookup_tensor, labels, mask)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
train(net, epoch_num, batch_size, trainset, trainloader)

testset = torch.utils.data.TensorDataset(lookup_tensor_test, labels_test, mask_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
test(net, batch_size, testset, testloader)
