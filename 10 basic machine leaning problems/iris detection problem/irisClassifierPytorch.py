import numpy as numpy, pandas as pd, torch, matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader

class DatasetClass(Dataset):
    def __init__(self, root=None):
        super().__init__()
        self.data = pd.read_csv(root, header=None)
        self.len = 125
    
    def __getitem__(self, idx):
        
        return self.data[idx]

    def __len__(self):
        return self.len

train_set = DatasetClass('iris.csv')
train_loader = DataLoader(train_set, 4, True)
test_set = DatasetClass('irisTest.csv')
test_loader = DataLoader(test_set, 4, True, num_workers=4)

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        h = self.sigmoid(x)
        return h
    
classifier = Classifier()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.6)

for epoch in range(5):
    print('ok')
    for i, data in enumerate(train_loader, 0):
        print('ok')
        inputs, classes = data
        print('ok')
        inputs, classes = Variable(inputs), Variable(classes)
        print('ok')

        y_pred = classifier(inputs)
        print('ok')
        loss = loss_function(y_pred, classes)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch)







