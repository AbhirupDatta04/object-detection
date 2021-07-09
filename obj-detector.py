import cv2 #for drawing the object detection
import torch
from torch.autograd import Variable
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio #to process images of video

#detection function with frame by frame detection 
def detection(frame,net,transform):                      #ssd neural network #transformations to have the right format as input of neural network
    h,w=frame.shape[0,1]
    frame_trans=transform(frame)[0]
    x=torch.from_numpy(frame_trans).permute(2,0,1) #RBG to GRB
    x=Variable(x.unsqueeze(0))                  #Batch for neural networks
    
   #Ready to feed the torch variables to neural network
   y=net(x)
   detections=y.data
   
    