import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform
import scipy.misc
import h5py
from matplotlib.pyplot import imshow
import keras.backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import os
import cv2

def identity_block(X,f,filters,stage,block):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    f1,f2,f3=filters
    X_shortcut=X

    X=Conv2D(filters=f1,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2a',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=f2,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=f3,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    X=Add()([X,X_shortcut])
    X=Activation('relu')(X)

    return X

def convolutional_block(X,f,filters,stage,block,s=2):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    f1,f2,f3=filters
    X_shortcut=X

    X=Conv2D(filters=f1,kernel_size=(1,1),strides=(s,s),name=conv_name_base+'2a',padding='valid',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=f2,kernel_size=(f,f),strides=(1,1),name=conv_name_base+'2b',padding='same',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=f3,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2c',padding='valid',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    X_shortcut=Conv2D(filters=f3,kernel_size=(1,1),strides=(s,s),name=conv_name_base+'1',padding='valid',
                      kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut=BatchNormalization(axis=3,name=bn_name_base+'1')(X_shortcut)

    X=Add()([X,X_shortcut])
    X=Activation('relu')(X)

    return X

def ResNetModel(input_shape=(224,224,3)):
    
    X_input=Input(input_shape)
    X=ZeroPadding2D((3,3))(X_input)
    
    #Stage1
    X=Conv2D(64,kernel_size=(7,7),strides=(2,2),name='conv1',
             kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='bn_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((3,3),strides=(2,2))(X)

    #Stage2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X=identity_block(X,3,[64,64,256],2,'b')
    X=identity_block(X,3,[64,64,256],2,'c')

    #Stage3
    X=convolutional_block(X,f=3,filters=[128,128,512],stage=3,block='a',s=2)
    X=identity_block(X,3,[128,128,512], 3, 'b')
    X=identity_block(X,3,[128,128,512],3,'c')
    X=identity_block(X,3,[128,128,512],3,'d')

    #Stage4
    X=convolutional_block(X,3,[256,256,1024],4,'a',s=2)
    X=identity_block(X,3,[256,256,1024],4,'b')
    X=identity_block(X,3,[256,256,1024],4,'c')
    X=identity_block(X,3,[256,256,1024],4,'d')
    X=identity_block(X,3,[256,256,1024],4,'e')
    X=identity_block(X,3,[256,256,1024],4,'f')

    #Stage5
    X=convolutional_block(X,3,[512,512,2048],5,'a',s=2)
    X=identity_block(X,3,[512,512,2048],5,'b')
    X=identity_block(X,3,[512,512,2048],5,'c')

    X=AveragePooling2D((2,2),name='avg_pool',padding='same')(X)
    X=Flatten()(X)
    #X=Dense(1000,activation='softmax',name='fc'+'1',kernel_initializer=glorot_uniform(seed=0))(X)
    X=Dense(30,activation='softmax',name='fc'+'6',kernel_initializer=glorot_uniform(seed=0))(X)

    model=Model(inputs=X_input,outputs=X,name='ResNetModel')

    return model
"""model=ResNetModel(input_shape=(224,224,3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

list=[]
img_path="F:\\Capestone Project\\FACE RECOGNITION SYSTEM\\Subject04"

for x in os.listdir(img_path):
    list.append(x)
p=[]
img_path1="F:\\Capestone Project\\FACE RECOGNITION SYSTEM\\Subject01"
for x in os.listdir(img_path1):
    p.append(x)
total=len(list)+len(p)
x=np.zeros((total,224,224,3))
b=np.zeros((1,total),dtype=np.int8)
for i in range(0,total):
    if(i<len(list)):
        img = img_path+'\\'+list[i]
    else:
        img = img_path1+'\\'+p[i-len(list)]
        b[0][i]=1
    img = image.load_img(img, target_size=(224, 224))
    x[i] = image.img_to_array(img)
    x[i] = np.expand_dims(x[i], axis=0)
    x[i]= preprocess_input(x[i])
    x[i]=abs(x[i])
#x=x/255

b= convert_to_one_hot(b, 2).T
model.evaluate(x)
#model.fit(x, b,epochs=4,batch_size=8)

#model.save('asdf.h5')"""








    

    

    
    

    
