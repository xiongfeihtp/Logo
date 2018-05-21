'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: settings_logo.py
@time: 2018/5/16 下午1:25
@desc: shanghaijiaotong university
'''
import os
import sys
import json
import platform
import datetime
import configparser

DB_CONFIG_PATH = r'D:\Users\zhouyuan346\PycharmProjects\zhouyuan2017\logo_similarity\case_parse_config_database.ini'
conf = configparser.ConfigParser()
conf.read(DB_CONFIG_PATH)

PG_DB_NAME = conf.get("pg_db", "PG_DB_NAME")
PG_USER_NAME = conf.get("pg_db", "PG_USER_NAME")
PG_PASSWORD = conf.get("pg_db", "PG_PASSWORD")
PG_HOST = conf.get("pg_db", "PG_HOST")
PG_PORT = conf.get("pg_db", "PG_PORT")
config = {"host": PG_HOST,
 "user": PG_USER_NAME,
 "password": PG_PASSWORD,
 "port": PG_PORT,
 "database": PG_DB_NAME
}
VGG_path = r'D:\Users\zhouyuan346\tensorflow-vgg'