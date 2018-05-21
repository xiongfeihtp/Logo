'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: task20180418_Siamese_res_net_ed1.py
@time: 2018/5/16 下午1:29
@desc: shanghaijiaotong university
'''
from keras import optimizers
from keras.layers import Dense, Input, Lambda, multiply
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
import os
import json
import numpy as np
import random
from itertools import combinations

os.chdir(r'D:\Users\zhouyuan346\PycharmProjects\zhouyuan2017\logo_similarity')


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def add_Dense_layer(base_model, nb_classes):
    X = base_model.output
    X = Dense(nb_classes, activation='softmax', name='dense_new')(X)
    model = Model(inputs=base_model.inputs, outputs=X, name='fine_tune_Res_net')
    return model


def freeze_layers(model, last_layers_num):
    for layer in model.layers[:last_layers_num]:
        layer.trainable = False


def Siamese_net(basic_model):
    input_image_1 = Input(shape=(224, 224, 3))
    input_image_2 = Input(shape=(224, 224, 3))

    encoded_image_1 = basic_model(input_image_1)
    encoded_image_2 = basic_model(input_image_2)


    # l1_distance_layer = Lambda(
    #     lambda tensors: K.abs(tensors[0] - tensors[1]))
    # l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # Same class or not prediction
    # prediction = Dense(units=1, activation='sigmoid')(l1_distance)
    cosine_similar = Lambda(lambda tensors: K.sum(multiply(tensors), axis=1, keepdims=True) / K.sqrt(
        K.sum(K.pow(tensors[0], 2), axis=1, keepdims=True) * K.sum(K.pow(tensors[1], 2), axis=1, keepdims=True)))
    prediction = cosine_similar([encoded_image_1, encoded_image_2])

    siamese_model = Model(inputs=[input_image_1, input_image_2], outputs=prediction, name='Siamese_net')
    return siamese_model


def get_train_X_Y(dataset_path, target_shape):
    train_path = os.path.join(dataset_path, 'train')
    validation_path = os.path.join(dataset_path, 'validation')
    train_dictionary = {}
    validation_dictionary = {}
    # First let's take care of the train alphabets
    for alphabet in os.listdir(train_path):
        alphabet_path = os.path.join(train_path, alphabet)

        current_alphabet_dictionary = os.listdir(alphabet_path)

        train_dictionary[alphabet] = current_alphabet_dictionary

    # Now it's time for the validation alphabets
    # for alphabet in os.listdir(validation_path):
    #     alphabet_path = os.path.join(validation_path, alphabet)
    #
    #     current_alphabet_dictionary = os.listdir(alphabet_path)
    #
    #     validation_dictionary[alphabet] = current_alphabet_dictionary

    pairs_label_1_ls = []
    pairs_label_0_ls = []
    NP_ratio = 2
    for pair in train_dictionary:
        pair_ls = [os.path.join(dataset_path, 'train', pair, i) for i in train_dictionary[pair]]
        pairs_label_1_ls.extend(list(combinations(pair_ls, 2)))
        other_pair = list(train_dictionary.keys())
        other_pair.remove(pair)
        selected_other_pair = random.sample(other_pair, NP_ratio)
        for choose1 in pair_ls:
            for pair_j in selected_other_pair:
                pair_j_ls = [os.path.join(dataset_path, 'train', pair_j, i) for i in train_dictionary[pair_j]]
                choose2 = random.sample(pair_j_ls, 1)
                pairs_label_0_ls.append((choose1, choose2))

    n_1 = len(pairs_label_1_ls)
    n_0 = len(pairs_label_0_ls)
    number_of_pairs = n_1 + n_0
    pairs_of_images = [np.zeros((number_of_pairs, target_shape[0], target_shape[1], 3)) for i in range(2)]
    labels = np.zeros((number_of_pairs, 1))
    index1 = 0
    for i_0, i_1 in pairs_label_1_ls:
        img1 = load_img(i_0, target_shape)
        img1 = img_to_array(img1, data_format='channels_last')
        pairs_of_images[0][index1] = img1
        img2 = load_img(i_1, target_shape)
        img2 = img_to_array(img2, data_format='channels_last')
        pairs_of_images[1][index1] = img2
        labels[index1] = 1
        index1 += 1

    for i_0, i_1 in pairs_label_1_ls:
        img1 = load_img(i_0, target_shape)
        img1 = img_to_array(img1, data_format='channels_last')
        pairs_of_images[0][index1] = img1
        img2 = load_img(i_1, target_shape)
        img2 = img_to_array(img2, data_format='channels_last')
        pairs_of_images[1][index1] = img2
        labels[index1] = 0
        index1 += 1

    train_indice = np.arange(0, number_of_pairs)
    np.random.shuffle(train_indice)

    pairs_of_images[0] = pairs_of_images[0][train_indice]
    pairs_of_images[1] = pairs_of_images[1][train_indice]
    labels = labels[train_indice]

    return pairs_of_images, labels


if __name__ == '__main__':
    if os.path.isfile('Siamese_net_res_net.json'):
        print('Siamese_net_res_net model_exist!')
        with open('Siamese_net_res_net.json', 'r') as f:
            model_config = json.load(f)
        Siamese_net1 = Model.from_config(model_config)
        Siamese_net1.load_weights('Siamese_net_res_net.h5')
    else:
        print('create Siamese net from Res_net')
        with open('Res_net_transfer_info.json', 'r') as f:
            model_config = json.load(f)
        model_new = Model.from_config(model_config)
        model_new.load_weights('Res_net_transfer_weights.h5')
        basic_model = Model(inputs=model_new.inputs, outputs=model_new.get_layer('flatten_1').output, name='basic_model')
        del model_new
        freeze_layers(basic_model, -16)
        # print(basic_model.summary())
        Siamese_net1 = Siamese_net(basic_model)

    print(Siamese_net1.summary())
    Adam1 = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    Siamese_net1.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                         optimizer=Adam1)

    dataset_path = r'D:\Users\zhouyuan346\Desktop\zhouyuan\logo资料\pic\pairs_example'
    target_shape = (224, 224)
    X_train, Y_train = get_train_X_Y(dataset_path, target_shape)
    batch_size = 16
    epochs = 10
    Siamese_net1.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    model_info = Siamese_net1.get_config()
    json_str = json.dumps(model_info)
    with open('Siamese_net_res_net.json', 'w+') as f:
        f.write(json_str)
    Siamese_net1.save_weights('Siamese_net_res_net.h5')



    # test = [np.zeros((1, target_shape[0], target_shape[1], 3)) for i in range(2)]
    # test[0][0] = X_train[0][1]
    # test[1][0] = X_train[1][1]
    #
    # Siamese_net1.predict(test)





