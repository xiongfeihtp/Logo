'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: task20180417_select_nowadays_logo_ed1.py
@time: 2018/5/16 下午1:28
@desc: shanghaijiaotong university
'''

import pandas as pd
import os

file_path_label_dict = {'picture_part1': r'/data/zhouyuan346/logo_datasets/picture_part1',
                        'picture_part2': r'/data/zhouyuan346/logo_datasets/picture_part2',
                        'picture_part3': r'/data/zhouyuan346/logo_datasets/picture_part3',
                        'text': r'/data/zhouyuan346/logo_datasets/logo_text_last_resized',
                        'pingan': r'/data/zhouyuan346/logo_datasets/logo_pingan_last_resized'}
logo_info = pd.read_csv('logo_info.csv', index_col=False, encoding='utf8')

ls = []
for label, path in file_path_label_dict.items():
    ls = ls + os.listdir(path)

all_img = pd.DataFrame({'img_id': ls})

all_img_info = pd.merge(left=all_img, right=logo_info, on='img_id', how='inner')
print('after 2000 logo num : %d' % len(all_img_info))
all_img_info.to('logo_info_have_data.csv', index=False, encoding='utf8')

# save_path = '/data/zhouyuan346/logo_datasets/logo_nowadays'
# if os.path.isdir(save_path):
#     pass
# else:
#     os.mkdir(save_path)




