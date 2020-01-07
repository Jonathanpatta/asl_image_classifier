from mainmodel import MyModel
from data import Train_loader,Test_loader,image_test

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from time import time


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


def train():

    device = torch.device('cuda')
    MODEL_PATH = "asl_classifier.pt"    
    BATCH_SIZE = 100
    EPOCHS = 3
    img_size = 128

    net = MyModel(img_size=128, no_of_classes=27).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCHS):
        imgno=0
        #net.zero_grad()
        for imgs,labels in Train_loader:
            labels = one_hot(labels,27)
            imgs,labels = imgs.to(device),labels.to(device)
            optimizer.zero_grad()            

            outputs = net(imgs)

            acc = 0
            for i,output in enumerate(outputs):
                if output.argmax()==labels[i].argmax():
                    acc+=1           

            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()

            imgno += 100
            print(f"Epoch:{epoch}  img:{imgno}  loss:{loss} acc:{acc}")
        torch.save(net.state_dict(), MODEL_PATH)

def predict():
    with torch.no_grad():
        device=torch.device('cpu')
        net = MyModel(img_size=128, no_of_classes=27).to(device)
        net.load_state_dict(torch.load(r"asl_classifier.pt", map_location=device))
        for j,(imgs,labels) in enumerate(Test_loader):
            labels = one_hot(labels,27)
            imgs,labels = imgs.to(device),labels.to(device)
            outputs = net(imgs)
            #print([output.argmax() for output in outputs])
            #print([label.argmax() for label in labels])
            acc = 0
            for i,output in enumerate(outputs):
                if output.argmax()==labels[i].argmax():
                    acc+=1
            print(f"acc:{acc}")
            if j%5==4:
                break
import cv2
def predict_img(img):
    
    #img = cv2.imread(r"D:\data\asl\significant-asl-sign-language-alphabet-dataset\Training Set\F\color_5_0002 (4).png",0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    with torch.no_grad():
        device=torch.device('cpu')
        net = MyModel(img_size=128, no_of_classes=27).to(device)
        net.load_state_dict(torch.load(r"asl_classifier.pt", map_location=device))
        return int(net(torch.Tensor(img).view(1,1,128,128)).argmax())


if __name__=='__main__':
    
    train()