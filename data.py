import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

root_folder = r"D:\data\asl\significant-asl-sign-language-alphabet-dataset\Training Set"

IMSIZE = 128

transformations = transforms.Compose([
                                    #transforms.RandomRotation(degrees=20),
                                    transforms.Grayscale(),
                                    transforms.Resize((IMSIZE,IMSIZE)),
                                    #transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
                                    transforms.ToTensor(),
                                ]
                                )

images = ImageFolder(root_folder,transform=transformations)
no_of_imgs = len(images.imgs)
image_train,image_test = torch.utils.data.random_split(images,[int(0.9*no_of_imgs),int(0.1*no_of_imgs)])

'''import matplotlib.pyplot as plt
import cv2'''







Train_loader = torch.utils.data.DataLoader(image_train,
                                          batch_size=100,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=10)

Test_loader = torch.utils.data.DataLoader(image_test,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=0)

print("loaded")

