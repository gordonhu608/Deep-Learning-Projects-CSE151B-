# Modified by Colin Wang, Weitang Liu
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

def prepare_model(device, args=None):
    # load model, criterion, optimizer, and learning rate scheduler
    
    model = get_model(args)
    model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=20, bias=True))
    model = model.to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
    lr_scheduler = 0.001

    return model, criterion, optimizer, lr_scheduler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args=None):
    
    #raise NotImplementedError()
    train_loader, val_loader, test_loader = dataloaders
    
    total_loss = []
    total_val_loss = []
    val_loss_min = 1000.0
    
    for epoch in range(7):
        
        train_loss = 0.0
        train_correct = 0
        model.train()
        for images, labels in train_loader:
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
            
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        total_loss.append(train_loss)
            
        print("Train loss = {}".format(train_loss))
        print("Train Accuracy =", train_correct / len(train_loader.dataset))
        print("Finished", epoch+1, "epochs of training")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:

                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                val_loss += loss.item()
            
        val_loss /= len(val_loader)
            
        total_val_loss.append(val_loss)
        print("Val loss = {}".format(val_loss))
        print("Val Accuracy =", correct / len(val_loader.dataset))
    
    return model # return the model with weight selected by best performance 

# add your own functions if necessary
def test_model(model, criterion, device, dataloaders, args=None):
    
    _, _, test_loader = dataloaders
    model.eval()
    test_count = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += criterion(data, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        #test_loss /= len(test_loader.dataset)
    print("Test Accuracy =", correct / len(test_loader.dataset))