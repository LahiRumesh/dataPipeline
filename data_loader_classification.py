import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image,ImageDraw
from torchvision import transforms
import xml.etree.ElementTree as ET


img_dir="Data"


class Detection_Loader(Dataset):
    def __init__(self,img_dir,transform=None):
        self.img_dir=img_dir
        self.transform=transform

        self.img_names=os.listdir(img_dir)
        self.img_names.sort()
        self.img_names=[os.path.join(img_dir,img_name) for img_name in self.img_names] 

        self.annotations_name=os.listdir(annotation_dir)
        self.annotations_name.sort()
        self.annotations_name=[os.path.join(annotation_dir,annotaion_name) for annotaion_name in self.annotations_name]

        #print(self.annotations_name)


    def __getitem__(self,idx):
        img_name=self.img_names[idx]
        img=Image.open(img_name)

        annotaion_name=self.annotations_name[idx]
        #print(annotaion_name)
        annotaion_tree=ET.parse(annotaion_name)
        bnd_bx=annotaion_tree.find("object").find("bndbox")

   
        x_max=float(bnd_bx.find('xmax').text)
        x_min=float(bnd_bx.find('xmin').text)
        y_max=float(bnd_bx.find('ymax').text)
        y_min=float(bnd_bx.find('ymin').text)

        w=x_max-x_min
        h=y_max-y_min
        x=float(x_min+w/2)
        y=float(y_min+h/2)
        #print(x_max,x_min,y_max,y_min)   

        x/=img.size[0]
        w/=img.size[0]
        y/=img.size[1]
        h/=img.size[1]
        bond_box=(x,y,w,h)    #order must be x y w h
        
        if self.transform:
            img=self.transform(img)
        
        bond_box=torch.tensor(bond_box)
        #print(img.shape)
        return img,bond_box

    def __len__(self):
        return len(self.img_names)
        
  
def unpack_bound(bndbox,img):
    x,y,w,h=tuple(bndbox)
    x*=img.size[0]
    w*=img.size[0]
    y*=img.size[1]
    h*=img.size[1]

    xmin=x-w/2
    xmax=x+w/2
    ymin=y-h/2
    ymax=y+h/2

    bndbox=[xmin,ymin,xmax,ymax]
    return bndbox

def show(batch,pred_bndbox=None):
    img,bndbox=batch
    print(img.shape)
    img=transforms.ToPILImage()(img)
    img=transforms.Resize(512,512)(img)
    draw=ImageDraw.Draw(img)
    bndbox=unpack_bound(bndbox,img)
    
    print(bndbox)
    draw.rectangle(bndbox)
    if pred_bndbox is not None:
        pred_bndbox=unpack_bound(pred_bndbox,img)
        draw.rectangle(pred_bndbox,outline=1000)
    img.show()

#random image transform for testing
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


dt=Detection_Loader(img_dir,annotation_dir,transform=transforms.ToTensor())

print(dt)
item=dt[0]
show(item)
