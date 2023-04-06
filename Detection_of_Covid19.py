#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
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


# In[2]:


#filtering the warnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[3]:


#Load the Dataset
NonCovid_Data = Path("C:/Users/rajee/Data_covid/Train/NonCovid/")
Covid_Data = Path("C:/Users/rajee/Data_covid/Train/Covid/")
Normal_Data = Path("C:/Users/rajee/Data_covid/Train/Normal/")


# In[4]:


NonCovid_PNG_Path = list(NonCovid_Data.glob("*.png"))
Covid_PNG_Path = list(Covid_Data.glob("*.png"))
Normal_PNG_Path = list(Normal_Data.glob("*.png"))


# In[5]:


print("NONCOVID:\n", NonCovid_PNG_Path[0:5])
print("---" * 20)
print("COVID:\n", Covid_PNG_Path[0:5])
print("---" * 20)
print("NORMAL:\n", Normal_PNG_Path[0:5])
print("---" * 20)


# In[6]:


Main_PNG_Path = NonCovid_PNG_Path + Covid_PNG_Path + Normal_PNG_Path


# In[7]:


PNG_All_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Main_PNG_Path))


# In[8]:


print("NonCovid:", PNG_All_Labels.count("NonCovid"))
print("Covid:", PNG_All_Labels.count("Covid"))
print("Normal:", PNG_All_Labels.count("Normal"))


# In[9]:


Main_PNG_Path_Series = pd.Series(Main_PNG_Path,name="PNG").astype(str)
PNG_All_Labels_Series = pd.Series(PNG_All_Labels,name="CATEGORY")


# In[10]:


Main_Data = pd.concat([Main_PNG_Path_Series,PNG_All_Labels_Series],axis=1)


# In[11]:


print(Main_Data.head(-1))


# In[12]:


print(Main_Data["CATEGORY"].value_counts())


# In[13]:


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


# In[14]:


Main_Data = Main_Data.sample(frac=1).reset_index(drop=True)


# In[15]:


print(Main_Data.head(-1))


# In[16]:


figure = plt.figure(figsize=(8,8))
sns.histplot(Main_Data["CATEGORY"])
plt.show()


# In[17]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][0])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][0])


# In[18]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][1])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][1])


# In[19]:


figure = plt.figure(figsize=(10,10))
x = plt.imread(Main_Data["PNG"][10578])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(Main_Data["CATEGORY"][10578])


# In[20]:


Train_Data,Test_Data = train_test_split(Main_Data,train_size=0.8,random_state=42,shuffle=True)


# In[21]:


print(Train_Data.shape)
print(Test_Data.shape)


# In[22]:


print(Train_Data.head(-1))
print(Test_Data.head(-1))


# In[23]:


Generator = ImageDataGenerator(rescale=1./255,
                              validation_split=0.1,
                               horizontal_flip=False,
                               featurewise_center=False,
                                    featurewise_std_normalization=False,
                               rotation_range=20,
                               zoom_range=0.2,
                               shear_range=0.2)


# In[24]:


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


# In[25]:


Train_IMG_Set = Generator.flow_from_dataframe(dataframe=Train_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             subset="training",
                                             batch_size=32)


# In[26]:


Validation_IMG_Set = Generator.flow_from_dataframe(dataframe=Train_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             subset="validation",
                                             batch_size=32)


# In[27]:


Test_Generator = ImageDataGenerator(rescale=1./255)
Test_IMG_Set = Test_Generator.flow_from_dataframe(dataframe=Test_Data,
                                             x_col="PNG",
                                             y_col="CATEGORY",
                                             color_mode="rgb",
                                             class_mode="categorical",
                                             batch_size=32)


# In[28]:


for data_batch,label_batch in Train_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[29]:


for data_batch,label_batch in Validation_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[30]:


for data_batch,label_batch in Test_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break


# In[31]:


print(Train_IMG_Set.class_indices)
print(Train_IMG_Set.classes[0:5])
print(Train_IMG_Set.image_shape)


# In[32]:


print(Validation_IMG_Set.class_indices)
print(Validation_IMG_Set.classes[0:5])
print(Validation_IMG_Set.image_shape)


# In[33]:


print(Test_IMG_Set.class_indices)
print(Test_IMG_Set.classes[0:5])
print(Test_IMG_Set.image_shape)


# In[34]:


#Creating the model
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


# In[35]:


keras.utils.plot_model(model,show_shapes=True)


# In[36]:


hist = model.fit(Train_IMG_Set, validation_data=Test_IMG_Set,steps_per_epoch=100,epochs=15)


# In[37]:


model_results = model.evaluate(Test_IMG_Set,verbose=False)
print("LOSS:  " + "%.4f" % model_results[0])
print("ACCURACY:  " + "%.2f" % model_results[1])


# In[38]:


model.save('detect.h5')


# In[39]:


plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()


# In[40]:


HistoryDict = hist.history

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1,len(val_losses)+1)


# In[41]:


plt.plot(epochs,val_losses,"k-",label="LOSS")
plt.plot(epochs,val_acc,"r",label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()


# In[42]:


New_Img_Path = "C:/Users/rajee/Data_covid/Train/Covid/images/covid_1.png"
IMG_Load = tf.keras.utils.load_img(New_Img_Path,target_size=(256,256))


# In[43]:


N_IMG_Array = tf.keras.utils.img_to_array(IMG_Load)


# In[44]:


print(N_IMG_Array.shape)


# In[45]:


N_IMG_Array = np.expand_dims(N_IMG_Array,axis=0)


# In[46]:


print(N_IMG_Array)


# In[47]:


pred = model.predict(N_IMG_Array)


# In[77]:


from keras.preprocessing import image
import tensorflow as tf
img = tf.keras.utils.load_img("C:/Users/rajee/Data_covid/Train/NonCovid/non_COVID (1).png",target_size=(256,256))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("detect.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("Covid")
else:
    print('NonCovid')


# In[49]:


from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

model = load_model('C:/Users/rajee/detect.h5')

ixs = [1, 2, 3, 4, 5,6]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

img = load_img('C:/Users/rajee/Data_covid/Train/Covid/images/covid_7.png', target_size=(256, 256))

img = img_to_array(img)

img = expand_dims(img, axis=0)

img = preprocess_input(img)

feature_maps = model.predict(img)

square = 1
for fmap in feature_maps:
    ix = 1
    for _ in range(square):
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
        ix += 1
    pyplot.show()


# In[50]:


from keras.preprocessing import image
import os
positive = 0
negative = 0
saved_model = load_model("detect.h5")
for dirname, _, filenames in os.walk('C:/Users/rajee/Data_covid/Test/Covid/'):
    for filename in filenames:
        img = tf.keras.utils.load_img(os.path.join(dirname, filename),target_size=(256,256))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        output = saved_model.predict(img)
        if output[0][0] > output[0][1]:
            print("Positive")
            positive+=1
        else:
            print('Negative')
            negative+=1
print("Total Covid Predicted ->",positive)
print("Total NonCovid Predicted ->",negative)


# In[52]:


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


# In[53]:


from keras.models import load_model
model = load_model('C:/Users/rajee/detect.h5')
Model_Test_Prediction = model.predict(Test_IMG_Set)


# In[54]:


Model_Test_Prediction = Model_Test_Prediction.argmax(axis=-1)


# In[55]:


print(Model_Test_Prediction)


# In[75]:


import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('C:/Users/rajee/detect.h5')

img = Image.open('C:/Users/rajee/Data_covid/Train/Covid/covid_1.png')
img = img.resize((256, 256))
img = np.asarray(img)
img = img / 255.0 

img = np.repeat(img[..., np.newaxis], 3, -1)

img = np.expand_dims(img, axis=0)

img = np.squeeze(img)

pred = model.predict(np.array([img]))

if np.where(pred < 0.5, 0, 1).sum() > 0:
    severity = 'Severe'
else:
    severity = 'Mild'


# In[76]:


print(pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




