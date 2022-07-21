#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf ## pip install tensorflow-gpu
import cv2 ### pip install opencv-python
## pip install opencv-contrib-python fullpackage
import os
import matplotlib.pyplot as plt ## pip install matlplotlib 
import numpy as np ## pip install


# In[2]:


Datadirectory = "D:/Placement_Preparations/Minor_Project/train/"  #Trainung Dataset


# In[3]:


Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"] #List of classes => exact name of folder 


# In[4]:


for category in Classes: 
    path = os.path.join(Datadirectory, category) ## // 
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        #backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) 
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)) 
        plt.show()
        break
    break


# # resize the img

# In[5]:


img_size = 224  #Image => 224x224

new_array= cv2.resize(img_array, (img_size,img_size)) 
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)) 
plt.show()


# In[6]:


new_array.shape


# # read all the image and converting them into array

# In[7]:


training_Data = [] ## data array

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category) 
        class_num = Classes.index(category) ## @ 1, ## Label 
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img)) 
                new_array= cv2.resize(img_array, (img_size,img_size)) 
                training_Data.append([new_array,class_num]) 
            except Exception as e:
                pass


# In[10]:


create_training_Data()


# In[11]:


print(len(training_Data))


# In[12]:


import random

random.shuffle(training_Data)


# In[13]:


x = [] # data/features
y = [] # Label
for features,label in training_Data:
    x.append(features)
    y.append(label)
    
x = np.array(x).reshape(-1, img_size, img_size, 3) # converting it into 4D


# In[14]:


x.shape


# # normalize the Data

# In[15]:


x = x.astype(np.float16)/255


# In[16]:


y[1000]


# In[17]:


y = np.array(y)


# In[18]:


y.shape


# # Deep Learning model for training - Transfer Learning

# In[2]:


import tensorflow as tf 
from tensorflow import keras

from tensorflow.keras import layers


# In[20]:


model = tf.keras.applications. MobileNetV2()## pre-trained model
model.summary()


# # Transfer Learning - Tuning, Weigth will start from last check point

# In[21]:


base_input = model.layers[0].input


# In[22]:


base_output = model.layers[-2].output


# In[23]:


base_output


# In[24]:


final_output = layers.Dense(128)(base_output) 
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output) 
final_output = layers. Activation('relu')(final_output) 
final_output = layers.Dense(7,activation = 'softmax')(final_output)


# In[25]:


final_output


# In[26]:


new_model = keras.Model(inputs = base_input, outputs = final_output)


# In[27]:


new_model.summary()


# In[ ]:


new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[ ]:


new_model.fit(x,y, epochs = 30 , batch_size = 10)


# In[ ]:


new_model.save('D:/Placement_Preparations/Minor_Project/my_model_2.h5')


# In[28]:


epoch=["1", "2", "3", "4", "5" , "6" , "7" , "8", "9" , "10" , "11", "12" , "13" , "14" , "15" , "16", "17" , "18" , "19" , "20" , "21" , "22" , "23" , "24" , "25" , "26" , "27" , "28" , "29" , "30"]
accuracy=[0.3896 , 0.4644 , 0.4902 , 0.5184 , 0.5359 , 0.5555 , 0.5720 , 0.6040 , 0.6144 , 0.6372 , 0.6544 , 0.6761 , 0.6903 , 0.7157 , 0.7387 , 0.7530 , 0.7693 , 0.7909 , 0.8040 , 0.8210 , 0.8364 , 0.8458 , 0.8607 , 0.8683 , 0.8743 , 0.8874 , 0.8899 , 0.8978 , 0.9027 , 0.9063]
loss=[1.5823 , 1.4002 , 1.3226 , 1.2517 , 1.2026 , 1.1485 , 1.1097 , 1.0474 , 1.0130 , 0.9581 , 0.9155 , 0.8553 , 0.8195 , 0.7607 , 0.7087 , 0.6635 , 0.6170 , 0.5658 , 0.5315 , 0.4885 , 0.4525 , 0.4260 , 0.3838 , 0.3698 , 0.3458 , 0.3159 , 0.3109 , 0.2962 , 0.2743 , 0.2689]

plt.plot (epoch, accuracy, label="Accuracy")

plt.plot (epoch, loss, label="Loss")

plt.xlabel('Epochs')

plt.ylabel('Accuracy & Loss')

plt.title ('Plotting Accuracy and Loss Graph for Trained Model')

plt.legend ()

plt.show()


# In[3]:


new_model= tf.keras.models.load_model('D:/Placement_Preparations/Minor_Project/my_model_2.h5')


# In[4]:


frame = cv2.imread("D:/Placement_Preparations/Minor_Project/download.jpg")


# In[5]:


frame.shape


# In[6]:


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# In[7]:


faceCascade = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[8]:


gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)


# In[9]:


faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for x,y,w,h in faces:
    roi_gray=gray[y:y+h, x:x+w]
    roi_color= frame [y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
    facess=faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex:ex + ew]


# In[10]:


plt.imshow (cv2.cvtColor (frame, cv2.COLOR_BGR2RGB))


# In[11]:


plt.imshow (cv2.cvtColor (face_roi, cv2.COLOR_BGR2RGB))


# In[12]:


final_image =cv2.resize(face_roi, (224, 224))
final_image = np.expand_dims(final_image, axis =0) ## need fourth dimension
final_image=final_image/255.0


# In[13]:


Predictions = new_model.predict(final_image)


# In[14]:


np.argmax(Predictions)


# In[16]:


import cv2,time

path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set the rectangle background to white
rectangle_bgr = (255, 255, 255)

#make a black image
img = np.zeros((500, 500))

#set some text
text = "Some text in a box!"

#get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

#set the text start position
text_offset_x = 10
text_offset_y= img.shape [0] - 25

#make the coords of the box with a small palding of two pixels 
box_coords= ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords [1], rectangle_bgr, cv2.FILLED) 
cv2.putText (img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture (0)

if not cap.isOpened(): 
    raise IOError("Cannot open webcam")

while True:
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew] # cropping the face

    final_image = cv2.resize(face_roi, (224,224)) 
    final_image = np.expand_dims(final_image,axis=0 ) ##a need fourth dimension

    final_image = final_image/255.0
    
    font= cv2.FONT_HERSHEY_SIMPLEX
    Predictions = new_model.predict(final_image)

    font_scale = 1.5
    font= cv2.FONT_HERSHEY_PLAIN

    if (np.argmax(Predictions)==0): 
        status= "Angry"
        x1, y1,w1, h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1+h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1+ int (w1/10), y1 + int (h1/2)), cv2. FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        
        cv2.putText(frame, status, (100, 150), font, 3,(0, 0, 255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)

    elif (np.argmax (Predictions)==1):
        status = "Disgust"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    elif (np.argmax (Predictions)==2):
        status = "Fear"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    elif (np.argmax (Predictions)==3):
        status = "Happy"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    elif (np.argmax (Predictions)==4):
        status = "Neutral"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    elif (np.argmax (Predictions)==5):
        status = "Sad"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    elif(np.argmax (Predictions)==6):
        status = "Surprise"
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
        # Add text 
        cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.0,255), 2)
        
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2, cv2.LINE_4)
        
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))
        
        time.sleep(1)
        
    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey (2) & 0xFF == ord ('q'): 
        break

cap.release()

cv2.destroyAllWindows()


# In[ ]:




