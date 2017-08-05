
# coding: utf-8

# In[2]:

import sys
from datetime import datetime
from os.path import join
from warnings import warn

import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[3]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[4]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'roomid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[5]:

feature_path = join(file_dir, 'room_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ### 添加基本特征 

# In[11]:

sample = extract_feature_count('roomid', 'hotel_roomid', train_df, sample)


# ## 历史是否为空

# In[10]:

# use_cols = ['uid', 'orderdate_lastord', 'hotelid', 'roomid', 'roomid_lastord']

# basic_feature = train_df[use_cols].drop_duplicates()

# if train_df.drop_duplicates(['uid', 'hotel_roomid']).shape[0] != basic_feature.shape[0]:
#     warn('[uid, basicroomid].shape[0] != basic_feature.shape[0]')

# sample = extract_lastord_is_nan(basic_feature, sample, 'roomid', 'roomid_lastord')

# sample = extract_is_lastord(basic_feature, sample, 'roomid', 'roomid_lastord')


# In[12]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

