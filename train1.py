from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from torchvision.models.resnet import Bottleneck
import torchvision.models as models
import math
import torch.utils.model_zoo as model_zoo
# from gensim.models import word2vec
# from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
# import torchmetrics 
     
from layer import HELLMEMECLASS






def train_model(model, criterion, device, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 
   
    # 

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            FN =0.0      

            FP =0.0

            TP =0.0  
            TN =0.0
            # Iterate over data.
            for inputs, labels ,contexts,faces in dataloaders[phase]:
                # print(inputs)
                # print(labels)
                # print(phase)
                # print(comment)
                # print()
               
               
                inputs = inputs.to(device)
                labels = labels.to(device)
                contexts = contexts.to(device)
                faces =  faces.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs,contexts,faces)
                    # print(outputs)
                    val, preds = torch.max(outputs, 1)
                    # print(preds)
                      
                    loss = criterion(outputs, labels)
                    # =======================
                    # print(labels.data)
                    zes=(torch.zeros(len(labels.data)).type(torch.LongTensor))#全0变量
                    zes=zes.to(device)
                    ons=(torch.ones(len(labels.data)).type(torch.LongTensor))#全1变量
                    ons=ons.to(device)
                    train_correct01 = ((preds==ons)&(labels.data==zes)).sum() #原标签为T，预测为 F 的总数

                    train_correct10 = ((preds==zes)&(labels.data==ons)).sum() #原标签为F，预测为 T 的总数

                    train_correct11 = ((preds==zes)&(labels.data==zes)).sum()
                    
                    train_correct00 = ((preds==ons)&(labels.data==ons)).sum()


                    # ======================================
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                FN += train_correct01.item()       

                FP += train_correct10.item()

                TP += train_correct11.item() 
                TN += train_correct00.item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
           
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
              print('TP'+str(TP)+' TN'+str(TN)+' FP'+str(FP)+' FN'+str(FN))
              print("accuracy"+str((TP +TN)/ (TP+TN+FP+FN)))
              print("precision:"+str(TP/ (TP+FP)))
              print("recall:"+str(TP/ (TP+FN)))
              print("F1:"+str(2*TP/ (2*TP+FP+FN)))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                


    plt.figure(0)
    plt.plot(range(1,num_epochs+1,1), np.array(train_loss), 'r-', label= "train loss") #relative global step
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.figure(1)
    plt.plot(range(1,num_epochs+1,1), np.array(valid_loss), 'b-', label= "eval loss") #--evaluate_during_training True 在啟用eval
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(),"model.pt")
    return model

def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    plt.figure(figsize=(18,9))

    with torch.no_grad():
     

            
        for i, (inputs, labels ,contexts,faces) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            contexts = contexts.to(device)
            faces =  faces.to(device)

            outputs = model(inputs,contexts,faces)
            a, preds = torch.max(outputs, 1)
            a[0].backward()
           
            for j in range(inputs.size()[0]):
                images_so_far += 1

                img_display = np.transpose(inputs.cpu().data[j].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
                plt.subplot(num_images//2,2,images_so_far),plt.imshow(img_display) #nrow,ncol,image_idx
                plt.title(f'predicted: {class_names[preds[j]]}')
                plt.savefig("test.jpg")
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    #原先Normalize是對每個channel個別做 減去mean, 再除上std
    inp1 = std * inp + mean

    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.imshow(inp1)
    if title is not None:
        plt.title(title)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

context_train = np.load("hell_npy_val/context_train1.npy")
context_test = np.load("hell_npy_val/context_test1.npy")
file_train = np.load("hell_npy_val/file_train1.npy")
file_test = np.load("hell_npy_val/file_test1.npy")
number_train = np.load("hell_npy_val/number_train1.npy")
number_test = np.load("hell_npy_val/number_test1.npy")
face_train = np.load("hell_npy_val/face_train1.npy")
face_test = np.load("hell_npy_val/face_test1.npy")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
preprocess = transforms.Compose([
            transforms.Resize((256,256) ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def default_loader(path):
    img_pil =  Image.open(path)
    # img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

#tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定義 image 的路徑
        self.images = file_train
        self.target = number_train
        self.context = context_train
        self.face = face_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        context = self.context[index]
        face = self.face[index]
        return img,target,context,face

    def __len__(self):
        return len(self.images)
class testset(Dataset):
    def __init__(self, loader=default_loader):
        #定義 image 的路徑
        self.images = file_test
        self.target = number_test
        self.context = context_test
        self.loader = loader
        self.face = face_test

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        context = self.context[index]
        face = self.face[index]
        return img,target,context,face

    def __len__(self):
        return len(self.images)
train_data  = trainset()
test_data = testset()




def main():
    num_workers = 2
    momentum = 0.9
    num_epochs = 10
    lr = 0.001
    batch_size = 32
    

    data_dir = './training'
    # trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])for x in ['train', 'val']}
    # print(train_data)
  
    dataloaders = {x: torch.utils.data.DataLoader(train_data if x=='train' else test_data, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(train_data) if x=='train' else len(test_data)  for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Get a batch of training data
    inputs, classes ,_,_= next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)


     #imshow(out, title=[class_names[x] for x in classes])
    
    
    pretrained_dict = models.resnet50(pretrained=True).state_dict()
    

    #imshow(out, title=[class_names[x] for x in classes])
    # pretrained_dict = torch.load("./f1_meme_model.pkl",map_location=torch.device(device))
    # for param in pretrained_dict.parameters():
    #     param.requires_grad = False
    # pretrained_dict = pretrained_dict.state_dict()


    model_ft = HELLMEMECLASS(Bottleneck, [3, 4, 6, 3])
    model_dict = model_ft.state_dict()
    print(set(model_dict.keys()) - set(pretrained_dict.keys()))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict if (k in model_dict and 'fc' not in k)}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model_ft.load_state_dict(model_dict)

    # for k,v in model_dict.items():
    #   print(k)
    # for (name, layer) in model_ft._modules.items():
    #     #iteration over outer layers
    #     print(f"{name}:\n{layer}\n")

    model_ft = model_ft.to(device)
    
    
    # model =======================================================================

    parameter_count = count_parameters(model_ft)
    print(f"#parameters:{parameter_count}")
    print(f"batch_size:{batch_size}")


    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(
            model_ft, 
            criterion, 
            device, 
            dataloaders, 
            dataset_sizes, 
            optimizer_ft, 
            exp_lr_scheduler,     
            num_epochs=num_epochs
    )

    # visualize_model(model_ft, device, dataloaders, l)


if __name__ == '__main__':
    main()
