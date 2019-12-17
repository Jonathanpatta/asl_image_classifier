import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn


cap = cv2.VideoCapture(0)

class Torchconvnet(nn.Module):
    def __init__(self):
        self.img_size = 128
        self.no_of_classes = 12
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(self.img_size,self.img_size).view(-1,1,self.img_size,self.img_size)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, self.no_of_classes) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


device = torch.device('cuda:0')
net = Torchconvnet()
net.load_state_dict(torch.load(r"C:\Users\Jonathan\Documents\pytorch\image_proc_project\fingers\data.pt", map_location=device))


def predict_img(net,img_path):
    import cv2
    #img_path = r"C:\Users\Jonathan\Downloads\banana.jpg"
    with torch.no_grad():
        #img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = img_path
        img = cv2.resize(img, (128,128))
        imgtensor = torch.Tensor(img).view(-1,128,128)
        imgtensor /= 255q
        #imgtensor = test_X[23]
        net_out = net(imgtensor.view(-1, 1, 128, 128))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out).tolist()

    #print(predicted_class,net_out[predicted_class],net_out.tolist())


#predict_img(net,r"C:\Users\Jonathan\Downloads\IMG_2032.jpg")
import time

t = time.time()
 
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        predict_img(net,gray)
        print(1/(time.time()-t))
    t = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()