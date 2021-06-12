#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from torchvision import models, transforms


# In[ ]:


gn=models.googlenet(pretrained=True).cuda()
#importing GoogleNet from models in torch vision  

img=transforms.ToTensor()

df = pd.DataFrame(columns=list(range(1000)))
target=pd.DataFrame(columns=['target'])
#getting the labels


# In[ ]:


for i in os.listdir('final_image_data'):
  for j in os.listdir('final_image_data/'+i):
    for k in os.listdir('final_image_data/'+i+'/'+j):
      im=Image.open('final_image_data/'+i+'/'+j+'/'+k)
      features=gn.forward(torch.autograd.Variable(img(im)).cuda().unsqueeze(0)).cpu().detach().numpy()
      df=df.append(pd.DataFrame(features))
      target=target.append(pd.DataFrame([int(j)])) 
#creating a feture vectors by feeding images to the GoogleNet  


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df2,target[0],train_size=0.8)
#test-train split


# In[ ]:


model=svm.SVC(kernel='linear',probability=True)
#SVM model


# In[ ]:


model.fit(x_train,y_train)
print("Model Accuracy=",model.score(x_test,y_test),model.score(x_train, y_train))

