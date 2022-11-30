import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func
from torchvision import models

class baseline(nn.Module):
    
    def __init__(self):
        super(baseline, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv1_normed = nn.BatchNorm2d(64)
        nn.init.xavier_normal_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv3.weight)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.conv4_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv4.weight)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc1_normed = nn.BatchNorm1d(128)
        nn.init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(in_features=128, out_features=20, bias=True)
        
    def forward(self, batch):
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = self.pool(batch)
        
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = self.avgpool(batch)
        
        batch = batch.view(-1, self.num_flat_features(batch))
        #print(batch.shape)
        
        batch = func.relu(self.fc1_dropout(self.fc1(batch)))
        batch = self.fc2(batch)
        
        return batch
    
    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

class custom(nn.Module):
    def __init__(self):
        super(custom, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv1_normed = nn.BatchNorm2d(64)
        nn.init.xavier_normal_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv3.weight)
        
        self.conv3_dropout = nn.Dropout(p=0.01)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv5_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv6_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv6.weight)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.conv4_normed = nn.BatchNorm2d(128)
        nn.init.xavier_normal_(self.conv4.weight)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.fc1_dropout = nn.Dropout(p=0.1)
        nn.init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(in_features=128, out_features=20, bias=True)
        
    def forward(self, batch):
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_dropout(self.conv3_normed(self.conv3(batch))))
        batch = func.relu(self.conv5_normed(self.conv5(batch)))
        batch = func.relu(self.conv6_normed(self.conv6(batch)))
        batch = self.pool(batch)
        
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = self.avgpool(batch)
        
        batch = batch.view(-1, self.num_flat_features(batch))
        #print(batch.shape)
        
        batch = func.relu(self.fc1_dropout(self.fc1(batch)))
        batch = self.fc2(batch)
        
        return batch
    
    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

class resnet(nn.Module):
    #implemented by using torchvision library
    pass

class vgg(nn.Module):
    #implemented by using torchvision library
    pass

def get_model(args):
    #model = baseline()
    #model = custom()
    #model = models.resnet18(pretrained=True)
    model = models.vgg16(pretrained=True)
    return model

