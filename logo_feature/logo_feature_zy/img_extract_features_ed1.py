'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: task20180328_img_extract_features_ed1.py
@time: 2018/5/16 下午1:27
@desc: shanghaijiaotong university
@Inform  : 提取特征
'''
import numpy as np
import cv2
import os
import sys
from sqlalchemy import create_engine
import pandas as pd
import base64
import psycopg2
import sqlalchemy.types
import tensorflow as tf
import datetime
import io
import settings_logo

def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flagImage = np.max(image,2)
    else:
        flagImage = image
    assert len(flagImage.shape) == 2

    row = np.where(np.max(flagImage,0) > threshold)[0]
    if row.size:
        cols = np.where(np.max(flagImage,1) > threshold)[0]
        image = image[cols[0]:cols[-1] + 1,row[0]:row[-1] + 1]
    else:
        image = image[:1,:1]
    return image


def encode_array(x):
    res = str(base64.b64encode(x),encoding='utf-8')
    return res

def decode_array(x):
    temp = base64.b64decode(bytes(x,encoding='utf-8'))
    res = np.fromstring(temp,np.float32)
    return res


def delete_table(database,table):
    db = eval(database)
    database_pg_d4 = db['plaws']
    conn = psycopg2.connect(**database_pg_d4)
    cur = conn.cursor()
    delete1 = ("delete from %s" % table)
    print('清空旧数据')
    cur.execute(delete1)
    conn.commit()
    cur.close()
    conn.close()


def logo_text_to_img(logo_text):
    try:
        temp = logo_text
        temp1 = base64.b64decode(temp)
        temp2 = np.fromstring(temp1, np.uint8)
        img = cv2.imdecode(temp2, cv2.IMREAD_COLOR)
        img = img[:, :, (2, 1, 0)]
        img = autocrop(img, 15)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    except:
        img = np.nan
    return img


def read_data(database,batchsize,row_start):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))
    #select the logo realtime
    date_0 = datetime.datetime.today().strftime('%Y-%m-%d')
    #datebase batch_size load
    ebd_qcc_logo_pic = pd.read_sql_query("""
    select *,row_number() over() from ebd_qcc_logo_pic where logo != '' and to_char(created_time,'yyyy-mm-dd') like '{2}' limit {0} offset {1}
    """.format(batchsize,row_start,date_0),engine_plaws)
    next_index = ebd_qcc_logo_pic['row_number'].max()
    ebd_qcc_logo_pic['logo'] = ebd_qcc_logo_pic.logo.apply(lambda x:logo_text_to_img(x))
    # plt.imshow(logo_text_to_img(ebd_qcc_logo_pic.logo[2]))
    # plt.imshow(ebd_qcc_logo_pic.logo[2])
    # ebd_qcc_logo_pic = pd.read_csv(r'D:\Users\zhouyuan346\Desktop\zhouyuan\2018-03-22\ebd_qcc_logo_pic.csv',sep=';')
    return ebd_qcc_logo_pic,next_index


def get_max_number(database,table):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))
    date_0 = datetime.datetime.today().strftime('%Y-%m-%d')
    number = pd.read_sql_query("""
    select count(*) from {0} where logo != '' and to_char(created_time,'yyyy-mm-dd') like '{1}'
    """.format(table,date_0),engine_plaws)
    number = number.get_values()[0,0]
    return number


def insert2pg(database,df,target_table,target_columns=None):
    db = eval(database)
    database_plaws = db['plaws']
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8', header=False, sep='\t', quoting=3)
output.seek(0)
    connection = psycopg2.connect(**database_plaws)
    cursor = connection.cursor()
    if target_columns is not None:
        cursor.copy_from(output, target_table, sep='\t', null='', columns=target_columns)
    else:
        cursor.copy_from(output, target_table, sep='\t', null='')
    connection.commit()
    cursor.close()
    connection.close()
    print('insert result finish!')

def execute(database,batchsize,row_start):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))

    feature_result_table = 'ebd_qcc_logo_feature'
    target_columns = ['tm_image_id','feature','created_by']
    #当天的logo
    n = get_max_number(database,'ebd_qcc_logo_pic')

    sys.path.append(settings_logo.VGG_path)
    import vgg16
    tf.reset_default_graph()
    # if row_start == 0:
    #     delete_table(database,feature_result_table)
    with tf.device('/cpu:0'):
        images = tf.placeholder("float", [batchsize, 224, 224, 3])
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
        vgg.build(images)
        with tf.Session() as sess:
            while row_start < n:
                print('batch %d' % (int(row_start / batchsize) + 1))
                ebd_qcc_logo_pic_batch, next_index = read_data(database, batchsize, row_start)
                na_index = ebd_qcc_logo_pic_batch.logo.isnull()
                #batch_size not enough or picture is destroy
                if len(na_index) < batchsize:
                    temp = pd.Series([False] * batchsize)
                    temp[:len(na_index)] = na_index
                    na_index = temp
                # plt.imshow(ebd_qcc_logo_pic_batch.logo[2])
                n_logo = len(ebd_qcc_logo_pic_batch)
                img_array = np.zeros((batchsize,224,224,3),dtype=np.uint8)
                for index,i in enumerate(ebd_qcc_logo_pic_batch.logo):
                    img_array[index] = i
                # plt.imshow(img_array[2])
                # feature = extract_img_VGG(img_array)
                feed_dict = {images: img_array}
                fc7 = sess.run(vgg.output_relu7, feed_dict=feed_dict)
                fc7[na_index] = np.nan
                # print(fc7.shape)
                feature = fc7
                ebd_qcc_logo_pic_batch['feature'] = None
                for i in range(n_logo):
                    ebd_qcc_logo_pic_batch.feature[i] = encode_array(feature[i])
                # list(ebd_qcc_logo_pic_batch.columns)
                ebd_qcc_logo_pic_batch = ebd_qcc_logo_pic_batch[target_columns]
                insert2pg(database,ebd_qcc_logo_pic_batch,feature_result_table,target_columns)
                # ebd_qcc_logo_pic_batch.to_sql(feature_result_table,con=engine_plaws,if_exists='append',index=False)

                row_start = next_index
    print('Finish')

if __name__ == '__main__':
    database = '{"plaws":%s,' \
               '"oula":{"host": "10.20.130.109","port":"7454","database":"pable", "password": "paic5678", "user": "pableopr"},' \
               '"ebd_pg_d4":{"host": "10.20.129.56","port":"7589","database":"d4paebd", "password": "paic5678", "user": "pierdata"}}' %(settings_logo.config)
    batchsize = 128
    row_start = 0
    execute(database,batchsize,row_start)

