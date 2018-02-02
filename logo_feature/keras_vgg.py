from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import json
from keras import backend as K
#墙外url失效的解决方案
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def moreCos(a,b):
    sum_fenzi = 0
    sum_fenmu = 1
    for i in range(len(a)):
        sum_fenzi += a[i]*b[i]
        sum_fenmu *= np.sqrt(a[i]**2 + b[i]**2 )
    return sum_fenzi/sum_fenmu
model = ResNet50(weights='imagenet',include_top=False)
path='./picture_place'
image_list=os.listdir(path)
train_img = []
for i in range(2):
    temp_img = image.load_img(path+'/'+image_list[i], target_size=(224, 224))
    temp_img = image.img_to_array(temp_img)
    train_img.append(temp_img)
# converting train images to array and applying mean subtraction processing
train_img = np.array(train_img)
train_img = preprocess_input(train_img)
x = preprocess_input(train_img)
output=model.predict(x)
#make the dir for visiable
vector_dic={}
for i in range(2):
    vector_dic[image_list[i]]=np.squeeze(output[i])
with open('./vector_dic','w') as f:
    json.dump(vector_dic,f)
#输出模型的第三层，并且根据训练和测试区分参数
# get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                   [model.layers[3].output])
# # output in test mode = 0
# layer_output = get_3rd_layer_output([x, 0])[0]
# # output in train mode = 1
# layer_output = get_3rd_layer_output([x, 1])[0]
