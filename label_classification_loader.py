import torch
import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import matplotlib.pyplot as plt

REBUILD_DATA=True 

parser=argparse.ArgumentParser()
parser.add_argument("--Data_Folder", type=str, default="Data", help="Training Images folder")
parser.add_argument("--IMAGE_SIZE", type=str, default=100, help="Image Size")
parser.add_argument("--TEST_SIZE", type=float, default=0.4, help="Testing data percentage")
args = parser.parse_args()


classes=os.listdir(args.Data_Folder)
full_path=list(os.path.join(args.Data_Folder,i) for i in classes)
with open('classes.txt', 'w') as f:
    for data in classes:
        f.write("%s\n" % data)

class Classification_Data():

    data_list=[]
    labels = {full_path[x]:x for x in range(len(full_path))}
    

    def generate_data(self):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path)
                    img=cv2.resize(img,(args.IMAGE_SIZE,args.IMAGE_SIZE))
                    self.data_list.append([np.array(img),self.labels[label]])

                except Exception as e:
                    pass

        np.random.shuffle(self.data_list)
        test_size=int(len(self.data_list)*args.TEST_SIZE)
        training_data,testing_data=self.data_list[test_size:],self.data_list[:test_size]
        np.save("training_data.npy",training_data) #Save training data to numpy array
        np.save("testing_data.npy",testing_data) #Save testing data to numpy array
      


if REBUILD_DATA:
    clas_data=Classification_Data()
    clas_data.generate_data()

training_data=np.load("training_data1.npy",allow_pickle=True)
#print(int(len(training_data)*0.5))
#training, test = training_data[:80,:], training_data[80:,:]


#print(training_data)

#for i in training_data:
#    print(i[1])

#print(training_data[7][1])

#plt.imshow(training_data[7][0])
#plt.show()
