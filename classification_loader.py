import torch
import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import matplotlib.pyplot as plt

REBUILD_DATA=False 

parser=argparse.ArgumentParser()
parser.add_argument("--Data_Folder", type=str, default="Data", help="Training Images folder")
parser.add_argument("--IMAGE_SIZE", type=str, default=100, help="Image Size")
args = parser.parse_args()


classes=os.listdir(args.Data_Folder)
full_path=list(os.path.join(args.Data_Folder,i) for i in classes)
with open('classes.txt', 'w') as f:
    for data in classes:
        f.write("%s\n" % data)

class Classification_Data():

    training_data=[]
    labels = {full_path[x]:x for x in range(len(full_path))}

    def generate_data(self):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path)
                    img=cv2.resize(img,(args.IMAGE_SIZE,args.IMAGE_SIZE))
                    self.training_data.append([np.array(img),np.eye(len(classes))[self.labels[label]]])

                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data) #Save data to numpy array
      


if REBUILD_DATA:
    clas_data=Classification_Data()
    clas_data.generate_data()

training_data=np.load("training_data.npy",allow_pickle=True)





plt.imshow(training_data[3][0])
plt.show()
