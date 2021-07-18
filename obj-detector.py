import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#detection function with frame by frame detection 
def detection(frame,net,transform):                      #ssd neural network #transformations to have the right format as input of neural network
    h,w=frame.shape[:2]
    frame_trans=transform(frame)[0]
    x=torch.from_numpy(frame_trans).permute(2,0,1) #RBG to GRB
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    detections = y.data
    scale=torch.Tensor([w,h,w,h])
    #detections=[batch,No.of classes,no. of occurence of class,(score,x0,y0,x1,y1)]
    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0]>0.6:
            pt=(detections[0,i,j,1:]*scale).numpy()#Normalization of scale of image
            cv2.rectangle(frame,(int (pt[0]),int (pt[1])) ,(int(pt[2]) ,int (pt[3])),(255,0,0),2)
            cv2.putText(frame, labelmap[i-1], (int (pt[0]),int (pt[1])), cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255),2,cv2.LINE_AA)
            j+=1
    return frame

#Creating SSD neural network from weights of pretrained neural network
net=build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=lambda storage,loc:storage))

#Creation of transformation
transform=BaseTransform(net.size, (104/256.0,117/256.0,123/256.0))


#Object detection
reader=imageio.get_reader('Adorable Golden Retrievers Puppies Play in a Field.mp4')
fps=reader.get_meta_data()['fps']
writer=imageio.get_writer('Output.mp4',fps=fps)
for i,frame in enumerate(reader):  
    frame=detection(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()

            
            
            
   
   
    