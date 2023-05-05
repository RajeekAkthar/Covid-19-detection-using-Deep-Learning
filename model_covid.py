#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D
from keras import models
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras import layers
import tensorflow as tf
import os
import os.path
from pathlib import Path
import cv2
import cvlib as cv
import keras
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.applications import VGG16
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.optimizers.legacy import Adam
import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler
from keras.utils import load_img,img_to_array
from keras.preprocessing import image
from tensorflow.keras.utils import load_img
import torch
import torchvision


# In[5]:


filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[6]:


NonCovid_Data = Path("C:/Users/rajee/Data_covid/Train/NonCovid/")
Covid_Data = Path("C:/Users/rajee/Data_covid/Train/Covid/")
Normal_Data = Path("C:/Users/rajee/Data_covid/Train/Normal/")


# In[7]:


NonCovid_PNG_Path = list(NonCovid_Data.glob("*.png"))
Covid_PNG_Path = list(Covid_Data.glob("*.png"))
Normal_PNG_Path = list(Normal_Data.glob("*.png"))


# In[8]:


print("NONCOVID:\n", NonCovid_PNG_Path[0:5])
print("---" * 20)
print("COVID:\n", Covid_PNG_Path[0:5])
print("---" * 20)
print("NORMAL:\n", Normal_PNG_Path[0:5])
print("---" * 20)


# In[9]:


Main_PNG_Path = NonCovid_PNG_Path + Covid_PNG_Path + Normal_PNG_Path


# In[10]:


PNG_All_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Main_PNG_Path))


# In[11]:


print("NonCovid:", PNG_All_Labels.count("NonCovid"))
print("Covid:", PNG_All_Labels.count("Covid"))
print("Normal:", PNG_All_Labels.count("Normal"))


# In[12]:


Main_PNG_Path_Series = pd.Series(Main_PNG_Path,name="PNG").astype(str)
PNG_All_Labels_Series = pd.Series(PNG_All_Labels,name="CATEGORY")


# In[13]:


Main_Data = pd.concat([Main_PNG_Path_Series,PNG_All_Labels_Series],axis=1)


# In[14]:


print(Main_Data.head(-1))


# In[15]:


print(Main_Data["CATEGORY"].value_counts())


# In[16]:


print(Main_Data["PNG"][1])
print(Main_Data["CATEGORY"][1])
print(Main_Data["PNG"][1398])
print(Main_Data["CATEGORY"][1398])
print(Main_Data["PNG"][9867])
print(Main_Data["CATEGORY"][9867])
print(Main_Data["PNG"][10675])
print(Main_Data["CATEGORY"][10675])
print(Main_Data["PNG"][11643])
print(Main_Data["CATEGORY"][11643])
print(Main_Data["PNG"][19258])
print(Main_Data["CATEGORY"][19258])
print(Main_Data["PNG"][20331])
print(Main_Data["CATEGORY"][20331])


# In[17]:


Main_Data = Main_Data.sample(frac=1).reset_index(drop=True)


# In[18]:


print(Main_Data.head(-1))


# In[19]:


figure = plt.figure(figsize=(8,8))
sns.histplot(Main_Data["CATEGORY"])
plt.show()


# In[20]:


Main_Data['CATEGORY'].value_counts().plot.pie(figsize=(9,4))
plt.show()


# In[21]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][0])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][0])


# In[22]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][1])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][1])


# In[23]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][10578])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][10578])


# In[24]:


Train_Data,Test_Data = train_test_split(Main_Data,train_size=0.8,random_state=42,shuffle=True)


# In[25]:


print(Train_Data.shape)


# In[26]:


print(Test_Data.shape)


# In[27]:


print(Train_Data.head(-1))


# In[28]:


print(Test_Data.head(-1))


# In[29]:


Generator = ImageDataGenerator(rescale=1./255,
                              validation_split=0.1,
                               horizontal_flip=False,
                               featurewise_center=False,
                                    featurewise_std_normalization=False,
                               rotation_range=20,
                               zoom_range=0.2,
                               shear_range=0.2)


# In[30]:


Example_IMG = Train_Data["PNG"][4]
IMG = tf.keras.utils.load_img(Example_IMG,target_size=(300,400))
Array_IMG = tf.keras.utils.img_to_array(IMG)
Array_IMG = Array_IMG.reshape((1,)+Array_IMG.shape)

i = 0
for BTCH in Generator.flow(Array_IMG,batch_size=1):
    plt.figure(i)
    IMG_Plot = plt.imshow(tf.keras.utils.array_to_img(BTCH[0]))
    i += 1
    if i % 6 == 0:
        break
plt.show()


# In[31]:


Train_IMG_Set = Generator.flow_from_dataframe(dataframe=Train_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             subset="training",
                                             batch_size=32)


# In[32]:


Validation_IMG_Set = Generator.flow_from_dataframe(dataframe=Train_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             subset="validation",
                                             batch_size=32)


# In[33]:


Test_Generator = ImageDataGenerator(rescale=1./255)


# In[34]:


Test_IMG_Set = Test_Generator.flow_from_dataframe(dataframe=Test_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             batch_size=32)


# In[35]:


for data_batch,label_batch in Train_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[36]:


for data_batch,label_batch in Validation_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[37]:


for data_batch,label_batch in Test_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[38]:


print(Train_IMG_Set.class_indices)
print(Train_IMG_Set.classes[0:5])
print(Train_IMG_Set.image_shape)


# In[39]:


print(Validation_IMG_Set.class_indices)
print(Validation_IMG_Set.classes[0:5])
print(Validation_IMG_Set.image_shape)


# In[40]:


print(Test_IMG_Set.class_indices)
print(Test_IMG_Set.classes[0:5])
print(Test_IMG_Set.image_shape)


# In[41]:


model = Sequential()
model.add(Conv2D(64,(3,3),activation = "relu", input_shape = (256,256,3)))
model.add(MaxPool2D())

model.add(Conv2D( 128, (3,3), activation = "relu"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D( 256,(3,3), activation = "relu"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D( 512,(3,3), activation = "relu"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.15))

model.add(Dense(3, activation = "softmax"))


model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


# In[42]:


keras.utils.plot_model(model,show_shapes=True)


# In[43]:


hist = model.fit(Train_IMG_Set, validation_data=Validation_IMG_Set,steps_per_epoch=128,epochs=30)


# In[44]:


model_results = model.evaluate(Test_IMG_Set,verbose=False)
print("LOSS:  " + "%.4f" % model_results[0])
print("ACCURACY:  " + "%.2f" % model_results[1])


# In[54]:


model.save('detect.h5')


# In[55]:


plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()


# In[56]:


HistoryDict = hist.history

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1,len(val_losses)+1)


# In[57]:


plt.plot(epochs,val_losses,"k-",label="LOSS")
plt.plot(epochs,val_acc,"r",label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()


# In[58]:
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

img_path = "C:/Users/rajee/Data_covid/Train/Covid/Covid_1.png"
img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
img = np.asarray(img)
plt.imshow(img)

img = np.expand_dims(img, axis=0)

# Modify the path and model name as per your requirements
saved_model = tf.keras.models.load_model('detect.h5')

# Modify the classes as per your requirements
classes = ['Covid', 'NonCovid', 'Normal']

output = saved_model.predict(img)
class_index = np.argmax(output, axis=1)[0]

print("Class:", classes[class_index])

# In[58]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 0, 2, 0, 1, 2, 1, 1, 2])

labels = ['COVID-19', 'Normal', 'Noncovid']
cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       ylabel='True label',
       xlabel='Predicted label')


fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()