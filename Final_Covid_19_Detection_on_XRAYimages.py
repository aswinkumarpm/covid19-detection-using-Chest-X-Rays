#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import shutil


# In[2]:


#Create the data for positive samples

FILE_PATH = "/home/aswin/Covid19-Detection-Using-Chest-X-Ray/covid-chestxray-dataset-master/metadata.csv"
IMAGE_PATH = "/home/aswin/Covid19-Detection-Using-Chest-X-Ray/covid-chestxray-dataset-master/images"


# In[3]:


df = pd.read_csv(FILE_PATH)
print("hi")
print(df.shape)


# In[4]:


print(df.head())


# # Load Datasets

# In[17]:


TARGET_DIR = "Dataset/Train/Covid"

if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("Covid FOlder Created")


# In[18]:


#Copy COVID-19 images with view point PA from Downloaded directory to Target Directory
cnt = 0
for(i,row) in df.iterrows():
    if row["finding"] == "Pneumonia/Viral/COVID-19" and row["view"] =="PA":
        filename = row["filename"]
        image_path = os.path.join(IMAGE_PATH,filename) #IMAGE_PATH + filename
        image_copy_path = os.path.join(TARGET_DIR,filename) #TARGET_DIR + filename
        shutil.copy2(image_path,image_copy_path) #Copy From IMAGE_PATH to TARGET_DIrectory
       # print("Moving Image",cnt)
        cnt+=1
# print(cnt)


# In[19]:


# Sampling of Images from Kaggle Data,As we have 143 around covid images so we will have 143 around normal images of xray

import random
KAGGLE_FILE_PATH = "/home/aswin/Covid19-Detection-Using-Chest-X-Ray/archive/chest_xray/train/NORMAL"
TARGET_NORMAL_DIR = "Dataset/Train/Normal"
if not os.path.exists(TARGET_NORMAL_DIR):
    os.mkdir(TARGET_NORMAL_DIR)
    print("Normal Folder Created")

image_names = os.listdir(KAGGLE_FILE_PATH) #COntains list of all image names
# images_names
random.shuffle(image_names) #it will randomly shuffle names in list

for i in range(144):
    image_name = image_names[i]
    image_path = os.path.join(KAGGLE_FILE_PATH,image_name)

    target_path = os.path.join(TARGET_NORMAL_DIR,image_name)
    shutil.copy2(image_path,target_path)
#     print("Moved",i)


# In[20]:


TRAIN_PATH = "Dataset/Train"
VAL_PATH = "Dataset/Val"


# # **Importing Required Libraries**

# In[129]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# # **Building Architecture**
#

# In[22]:


model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))

model.compile(loss=keras.losses.binary_crossentropy,optimizer = "adam",metrics=["accuracy"])


# In[23]:


model.summary()


# # **Train From Scratch**

# ## *Data Augmentation*

# In[24]:


train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)
test_dataset = image.ImageDataGenerator(rescale = 1./255)

print(train_datagen, test_dataset, "hagshgasjhsa")
# In[26]:


train_generator = train_datagen.flow_from_directory(
    'Dataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)
print(train_generator, "This is  train generator")


# In[27]:


print(train_generator.class_indices)


# In[28]:


validation_generator = test_dataset.flow_from_directory(
    'Dataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)
print(validation_generator,  "This is  valid generator")


# # **Fit The Model**

# In[29]:


hist = model.fit_generator(
    train_generator,
    steps_per_epoch = 8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2
)

print(hist)

# ## *Loss is very less and accuracy is on point*

# In[30]:


model.save("Detection_Covid_19.h5")


# In[126]:


# model.evaluate_generator(train_generator)


# In[127]:


# model.evaluate_generator(validation_generator)


# # **Test Images**

# In[33]:


model = load_model("Detection_Covid_19.h5")


# In[34]:


import os


# In[35]:


print(train_generator.class_indices)


# # **Confusion Matrix**

# In[36]:


y_actual = []
y_test = []


# In[37]:
print(os.listdir("./Dataset/Val/Normal"))

for i in os.listdir("./Dataset/Val/Normal"):
  img = image.load_img("./Dataset/Val/Normal/"+i,target_size=(224,224))
  print("aswin")
  img = image.img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(1)

print(y_actual, y_test, "for normal")

# In[38]:


for i in os.listdir("./Dataset/Val/Covid"):
  img = image.load_img("./Dataset/Val/Covid/"+i,target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(0)
print(y_actual, y_test, "for covid")


# In[39]:


y_actual = np.array(y_actual)
y_test = np.array(y_test)
print(y_actual, y_test, "After Conversion")



# In[40]:


from sklearn.metrics import confusion_matrix


# In[41]:


cm = confusion_matrix(y_actual,y_test)
print(cm)


# In[45]:


import seaborn as sns


# In[130]:


# sns.heatmap(cm,cmap = "plasma" , annot=True)


# In[104]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
class_names = ["Covid-19","Normal"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="plasma"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm, "the cm value") 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## ***Confusion Matrix***

# In[111]:


plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix for Covid-19 Detection',cmap="plasma")


# # **List all data in history**

# In[107]:



history = hist
print(history.history.keys())


# # **Summarize history for accuracy**

# In[108]:



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # **Summarize history for loss**

# In[109]:



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # **Predictions from X-Ray Images**

# In[76]:


import numpy as np
# from google.colab.patches import cv2_imshow
import cv2
from keras.preprocessing import image
xtest_image = image.load_img('Dataset/Prediction/ryct.2020200034.fig5-day7.jpeg', target_size = (224, 224))
xtest_image = image.img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis = 0)
results = model.predict_classes(xtest_image)
# training_set.class_indices
imggg = cv2.imread('Dataset/Prediction/ryct.2020200034.fig5-day7.jpeg')
print("This Xray Image is of positive covid-19 patient")
imggg = np.array(imggg)
imggg = cv2.resize(imggg,(400,400))
plt.imshow(imggg)
# cv2_imshow(imggg)
# print(results)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model : ",prediction)


# In[80]:


import numpy as np
# from google.colab.patches import cv2_imshow
from keras.preprocessing import image
xtest_image = image.load_img('Dataset/Prediction/NORMAL2-IM-0354-0001.jpeg', target_size = (224, 224))
xtest_image = image.img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis = 0)
results = model.predict_classes(xtest_image)
# training_set.class_indices

imggg = cv2.imread('Dataset/Prediction/NORMAL2-IM-0354-0001.jpeg')
print("This Xray Image is of Negative covid-19 patient")
imggg = np.array(imggg)
imggg = cv2.resize(imggg,(400,400))

plt.imshow(imggg)
# cv2_imshow(imggg)
# print(results)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model : ",prediction)
