import torch
import os
import numpy as np
from tqdm import tqdm
import cv2



data_folder='Data/test_data'
classes=os.listdir(data_folder)
#print(len(classes))
IMG_SIZE=50
#for i in classes:
#    full_path.append(os.path.join(data_folder,i))


full_path=list(os.path.join(data_folder,i) for i in classes)
REBUILD_DATA=True 

class Classification_Data():

    training_data=[]
    labels = {full_path[x]:x for x in range(len(full_path))}

    def generate_train_data(self):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path)
                    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(len(classes))[self.labels[label]]])

                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data) #Save numpy array of Data 
      

cloder=Classification_Data()
cloder.generate_train_data()
'''
if REBUILD_DATA:
    dogvscat=Classification_DataLoader()
    dogvscat.make_train_data()


training_data=np.load("training_data.npy",allow_pickle=True)
#print(len(training_data))


import matplotlib.pyplot as plt

plt.imshow(training_data[3][0])
plt.show()
'''