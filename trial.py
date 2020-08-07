from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
device = torch.device("cpu")
#model.cuda()
model.to(device)
model.eval()
classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor



def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = 'video19.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))



new_car_id=0
arr_carid = []
arr_new_car_id=[]

new_truck_id=0
arr_truckid = []
arr_new_truck_id=[]

new_person_id=0
arr_personid = []
arr_new_person_id=[]

new_tl_id=0
arr_tlid = []
arr_new_tl_id=[]


new_bus_id=0
arr_busid = []
arr_new_bus_id=[]


frames = 0
counter =1
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #frames_arr.append(frame)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            cls = classes[int(cls_pred)]
            #print(int(obj_id))

           
            if cls == "car":
                

                if int(obj_id) not in arr_carid:
                    new_car_id+=1
                    arr_new_car_id.append(new_car_id)
                    arr_carid.append(int(obj_id))

                for i in range(len(arr_new_car_id)):
                    a = arr_carid[i]                   
                    if obj_id == a:
                        counter = arr_new_car_id[i]

                color = colors[0]
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                #color = colors[int(obj_id) % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(counter)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)


            if cls == "truck":
                

                if int(obj_id) not in arr_truckid:
                    new_truck_id+=1
                    arr_new_truck_id.append(new_truck_id)
                    arr_truckid.append(int(obj_id))
                                   
                for i in range(len(arr_new_truck_id)):
                    a = arr_truckid[i]                   
                    if obj_id == a:
                        counter = arr_new_truck_id[i]

                color = colors[1]
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                #color = colors[int(obj_id) % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(counter)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)


            if cls == "person":
                

                if int(obj_id) not in arr_personid:
                    new_person_id+=1
                    arr_new_person_id.append(new_person_id)
                    arr_personid.append(int(obj_id))
                                   
                for i in range(len(arr_new_person_id)):
                    a = arr_personid[i]                   
                    if obj_id == a:
                        counter = arr_new_person_id[i]

                color = colors[2]
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                #color = colors[int(obj_id) % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(counter)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)


            if cls == "traffic light":
                

                if int(obj_id) not in arr_tlid:
                    new_tl_id+=1
                    arr_new_tl_id.append(new_tl_id)
                    arr_tlid.append(int(obj_id))
                                   
                for i in range(len(arr_new_tl_id)):
                    a = arr_tlid[i]                   
                    if obj_id == a:
                        counter = arr_new_tl_id[i]

                color = colors[3]
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                #color = colors[int(obj_id) % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(counter)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)



            if cls == "bus":
                

                if int(obj_id) not in arr_busid:
                    new_bus_id+=1
                    arr_new_bus_id.append(new_bus_id)
                    arr_busid.append(int(obj_id))
                                   
                for i in range(len(arr_new_bus_id)):
                    a = arr_busid[i]                   
                    if obj_id == a:
                        counter = arr_new_bus_id[i]

                color = colors[4]
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                #color = colors[int(obj_id) % len(colors)]
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(counter)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                
                
                
           
                
              

    cv2.imshow('Stream', frame)

    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
