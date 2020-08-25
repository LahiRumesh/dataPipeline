import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.onnx as torch_onnx
from torch.autograd import Variable

REBUILD_DATA=False#rearrange training and testing numpy data sets
SAVE_DATA=True

parser=argparse.ArgumentParser()
parser.add_argument("--Data_Folder", type=str, default="../Data/pre_trained", help="Training Images folder")
parser.add_argument("--IMAGE_SIZE", type=str, default=224, help="Image Size")
parser.add_argument("--TEST_SIZE", type=float, default=0.1, help="Testing data percentage")
args = parser.parse_args()

classes=os.listdir(args.Data_Folder)
print(len(classes))
full_path=list(os.path.join(args.Data_Folder,i) for i in classes)
with open('classes.txt', 'w') as f:
    for i,data in enumerate(classes):
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
        if SAVE_DATA:
            np.save("training_data3.npy",training_data) #Save training data to numpy array
            np.save("testing_data3.npy",testing_data) #Save testing data to numpy array

        
        return training_data,testing_data
 
#if REBUILD_DATA:
#    clas_data=Classification_Data()
#    clas_data.generate_data()

classify_data=Classification_Data()
training_data,testing_data=classify_data.generate_data()


#training_data=np.load("training_data.npy",allow_pickle=True)
#testing_data=np.load("testing_data.npy",allow_pickle=True)

class Classification_DATASET(Dataset):
    def __init__(self,data_set,transform=None):
        self.data_set=data_set
        self.transform=transform
        #self.labels=labels

    def __len__(self):
        return(len(self.data_set))

    def __getitem__(self,idx):
        data = self.data_set[idx][0]
        data=data.astype(np.float32).reshape(3,224,224)

        if self.transform:
            data=self.transform(data)
            return (data,self.data_set[idx][1])
        else:
            return (data,self.data_set[idx][1])

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


train_data = Classification_DATASET(training_data)
test_data = Classification_DATASET(testing_data)

train_loader = DataLoader(train_data, batch_size=4, shuffle=False)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

#define NN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(classes))

model = model_ft.to(device)
learning_rate = 0.1
num_epochs = 20

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        print("Label Shape=",labels.shape)
        # Forward pass
        outputs = model(images)
        print("output Shape=",outputs.shape)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(2 * correct / total))

# Save the model checkpoint
input_shape = (3, 224, 224)
model_onnx_path = "test_model.onnx"
#model = Model()
#model.train(False)

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
print("Export of torch_model.onnx complete!")
#torch.save(model.state_dict(), 'model.ckpt')


