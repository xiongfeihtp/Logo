'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: task20180328_img_compare_similarity_ed1.py
@time: 2018/5/16 下午1:25
@desc: shanghaijiaotong university
# @Inform  : 比较相似性
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

def encode_array(x):
    res = str(base64.b64encode(x),encoding='utf-8')
    return res

def decode_array(x):
    temp = base64.b64decode(bytes(x,encoding='utf-8'))
    res = np.fromstring(temp,np.float32)
    return res

def cosine_dist(x,y):
    try:
        num = np.dot(x,y)
        denum = np.linalg.norm(x) * np.linalg.norm(y)
        res = num / denum
    except:
        res = np.nan
    return res


def load_other_feature(database,compare_size,row_start):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))

    # ebd_qcc_logo_feature_part = pd.read_sql_query("""
    # select * from ebd_qcc_logo_feature limit {0} offset {1}
    # """.format(compare_size,row_start),engine_plaws)

#     ebd_qcc_logo_feature_part = pd.read_sql_query("""
#         select A.*,B.company_name,B.int_cls,B.status,row_number() over() from ebd_qcc_logo_feature A,
#          (select name, id, company_name, int_cls, status from ebd_qcc_t_tm_info where company_name in
# ('上海友玩网络科技有限公司','平安壹钱包电子商务有限公司','上海陆家嘴国际金融资产交易市场股份有限公司','富登投资信用担保有限公司','平安大华基金管理有限公司',
# '平安国际融资租赁有限公司','平安证券有限责任公司','上海陆金所咨询有限公司','深圳德诚物业服务有限公司','深圳平安金融科技咨询有限公司','深圳市信安投资咨询有限公司',
# '平安付智能技术有限公司','深圳平安商用置业投资有限公司','平安惠普企业管理有限公司','中国平安保险(集团)股份有限公司')) B
#         where A.id != B.id
#          limit {0} offset {1}
#         """.format(compare_size, row_start), engine_plaws)
    ebd_qcc_logo_feature_part = pd.read_sql_query("""
          select A.tm_image_id, A.id,A.feature,B.name,B.company_name,B.int_cls,B.status,row_number() over() from ebd_qcc_logo_feature A left join
(select name, id, company_name, int_cls, status from ebd_qcc_t_tm_info) B 
on A.tm_image_id = B.id
        limit {0} offset {1}
        """.format(compare_size, row_start), engine_plaws)
    #bytes decode
    ebd_qcc_logo_feature_part.feature = ebd_qcc_logo_feature_part.feature.apply(lambda x:decode_array(x))
    next_index = ebd_qcc_logo_feature_part['row_number'].max()
    return ebd_qcc_logo_feature_part,next_index

def load_pingan_logo(database):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))
    ebd_qcc_pingan_logo = pd.read_sql_query("""
    select A.tm_image_id, A.id,A.feature,B.name,B.company_name,B.int_cls,B.status from ebd_qcc_logo_feature A inner join
(select name, id, company_name, int_cls, status from ebd_qcc_t_tm_info where company_name in 
('上海友玩网络科技有限公司','平安壹钱包电子商务有限公司','上海陆家嘴国际金融资产交易市场股份有限公司','富登投资信用担保有限公司','平安大华基金管理有限公司',
'平安国际融资租赁有限公司','平安证券有限责任公司','上海陆金所咨询有限公司','深圳德诚物业服务有限公司','深圳平安金融科技咨询有限公司','深圳市信安投资咨询有限公司',
'平安付智能技术有限公司','深圳平安商用置业投资有限公司','平安惠普企业管理有限公司','中国平安保险(集团)股份有限公司')) B 
on A.tm_image_id = B.id
    """,engine_plaws)
    ebd_qcc_pingan_logo.feature = ebd_qcc_pingan_logo.feature.apply(lambda x: decode_array(x))
    return ebd_qcc_pingan_logo


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

def compare_feature(database,compare_size,row_start):
    db = eval(database)
    database_plaws = db['plaws']
    engine_plaws = create_engine(''.join(
        ['postgresql+psycopg2://', database_plaws['user'], ':', database_plaws['password'], '@',
         database_plaws['host'], ':', str(database_plaws['port']), '/', database_plaws['database']]))

    n_all_other = pd.read_sql_query("""
    select count(*) from ebd_qcc_logo_feature
    """,engine_plaws).get_values()[0,0]


    result_table = 'ebd_qcc_logo_similarity_result'
    ebd_qcc_pingan_logo = load_pingan_logo(database)
#     ebd_qcc_pingan_logo = pd.read_sql_query("""
#           select A.tm_image_id, A.id,A.feature,B.name,B.company_name,B.int_cls,B.status,row_number() over() from ebd_qcc_logo_feature A left join
# (select name, id, company_name, int_cls, status from ebd_qcc_t_tm_info) B
# on A.tm_image_id = B.id
#         limit {0} offset {1}
#         """.format(20, 190), engine_plaws)
#     ebd_qcc_pingan_logo.feature = ebd_qcc_pingan_logo.feature.apply(lambda x:decode_array(x))

    # res_col = ['my_id','my_company_name','my_brand_name','my_field','target_id','target_company_name','target_brand_name','target_field','similarity','type','created_by','created_time'],'create_by','date_create','update_by','date_update'
    res_col = ['my_id','my_company_name','my_brand_name','my_field','target_id','target_company_name','target_brand_name','target_field','similarity','type']
   #pingan logo
    for i in range(len(ebd_qcc_pingan_logo)):
        res_temp = pd.DataFrame()
        print('scan similarity for the %d th pingan logo...' %i)
        pingan_i = ebd_qcc_pingan_logo.iloc[i]
        # compare_size = 200
        row_start = 0
        while row_start < n_all_other:
            #batch 200
            other_feature_part,next_index = load_other_feature(database,compare_size,row_start)
            #filter
            index_other = ~(other_feature_part.tm_image_id.isin(ebd_qcc_pingan_logo.tm_image_id))
            other_feature_part = other_feature_part[index_other]
            other_feature_part['similarity'] = other_feature_part.feature.apply(lambda x: cosine_dist(x,pingan_i.feature))
            #select the similarity other logo set
            res_part = other_feature_part.loc[other_feature_part.similarity > 0.85]
            res_part['my_id'] = pingan_i.tm_image_id
            res_part['my_company_name'] = pingan_i.company_name
            res_part['my_brand_name'] = pingan_i.name
            res_part['my_field'] = pingan_i.int_cls
            res_part['target_id'] = res_part['tm_image_id']
            res_part['target_company_name'] = res_part['company_name']
            res_part['target_brand_name'] = res_part['name']
            res_part['target_field'] = res_part['int_cls']
            res_part['type'] = 'VGG'
            res_part = res_part[res_col]
            row_start = next_index
            res_temp = pd.concat([res_temp,res_part],ignore_index=True)
        res_temp['similarity'] = res_temp['similarity'].apply(lambda x:round(float(x),3)).astype(str)
        #database input
        insert2pg(database,res_temp,result_table,res_col)
            # res_part.to_sql('ebd_qcc_logo_similarity_result',con=engine_plaws,if_exist='append',index=False)
    print('finish!')

if __name__ == '__main__':
    database = '{"plaws":%s,' \
               '"oula":{"host": "10.20.130.109","port":"7454","database":"pable", "password": "paic5678", "user": "pableopr"},' \
               '"ebd_pg_d4":{"host": "10.20.129.56","port":"7589","database":"d4paebd", "password": "paic5678", "user": "pierdata"}}' %(settings_logo.config)
    compare_size = 200
    row_start = 0
    compare_feature(database,compare_size,row_start)


