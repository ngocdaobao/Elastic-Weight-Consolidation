import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class Model(nn.Module):
    def __init__(self, in_dim, out_dim ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
                nn.Linear(self.in_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, self.out_dim),
            )
    def forward(self, x):
        y = self.layers(x)
        return y

class EWC(object):
    def __init__(self, model: Model, imgset):
        self.model = model
        self.imgset = imgset
        # Params for computing first order derivatives
        self.params = [p for _,p in self.model.named_parameters()]
        self.means = []
        for p in deepcopy(self.params):
            self.means.append(p)
    def Fisher_matrix(self, num_samples):
        self.FIM = []
        for i in range(len(self.params)):
            self.FIM.append(torch.tensor(np.zeros(list(self.params[i].shape))))

        self.model.eval()

        for _ in range(num_samples):
            self.model.zero_grad()
            #select random input
            idx = np.random.randint(len(self.imgset))
            x,y = self.imgset[idx]
            x = x[0].view(-1)
            logit = self.model(x)
            prob = torch.log_softmax(logit, dim=-1)
            score_function = F.nll_loss(prob.reshape(1,prob.shape[0]),torch.tensor([y]))
            score_function.backward()

            for i,p in enumerate(self.params):
                self.FIM[i] += p.grad**2
        self.FIM = [f/(num_samples) for f in self.FIM]

    def update_ewc(self, lamb, temp_model):
        loss = 0
        for i,p in enumerate(temp_model.parameters()):
            loss += (0.5*lamb*self.FIM[i]*(p-self.means[i])**2).sum()
        return loss

def sgd_train(model, dataloader, optimizer, criterion):
    model.train()
    for data, label in dataloader:
        img = data[0].view(data[0].shape[0],-1)
        logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def ewc_train(model, dataloader, optimizer, criterion, ewc_list, lamb):
    model.train()
    for data, label in dataloader:
        img = data[0].view(data[0].shape[0], -1)
        logits = model(img)
        loss = criterion(logits,label) + sum(ewc.update_ewc(lamb+i*200,model) for i,ewc in enumerate(ewc_list))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def get_accuracy(prob, label):
    predict = torch.argmax(prob, dim =1)
    acc = (predict == label).sum()/label.shape[0]
    return acc

def test(model, dataloader):
    acc = []
    model.eval()
    for data, label in dataloader:
        img = data[0].view(data[0].shape[0],-1)
        img = img
        label = label
        logits = model(img)

        prob = torch.softmax(logits, dim = 1)
        acc_score = get_accuracy(prob, label)
        acc.append(acc_score)
    return sum(acc)/len(dataloader)

