
# coding: utf-8

# In[1]:

import sys
from datetime import datetime
from os.path import join
from warnings import warn

import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[2]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', 'train')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[3]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'hotel_roomid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[4]:

feature_path = join(file_dir, 'hotel_room_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ## 添加基本特征 

# In[5]:

# sample = add_column(train_df, sample, 'hotel_roomid', 'room_30days_ordnumratio')

# sample = add_column(train_df, sample, 'hotel_roomid', 'room_30days_realratio')


# In[6]:

# train_df['price_real'] = train_df['price_deduct'] + train_df['returnvalue']


# In[7]:

# add_cols = ['basic_minarea', 'basic_maxarea', 'rank', 'roomservice_1']
# add_cols = []


# In[8]:

# for col in add_cols:
#     sample = add_column(train_df, sample, 'hotel_roomid', col)


# In[9]:

# get_corr(train_df, sample, 'hotel_roomid')


# In[10]:

serivice_cols = ['roomservice_%d' % x for x in range(1, 9)]


# In[11]:

tag_cols = ['roomtag_%d' % x for x in range(1, 5)]


# In[12]:

count_cols = serivice_cols + tag_cols


# In[13]:

for col in count_cols:
    sample = extract_feature_count('hotel_roomid', col, train_df, sample)


# In[14]:

# train_df.loc[(train_df.orderid==94304)&(train_df.basicroomid==25979)]


# ## 历史价格统计特征

# In[15]:

name_fmt = '{}_diff_{}'.format('hotel_roomid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[16]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[17]:

price_describe = ['mean', 'max']


# In[18]:

sample = extract_value_describe_feature('hotel_roomid', price_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('hotel_roomid', hotel_minprice_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('hotel_roomid', basic_minprice_diff_name, train_df, sample, price_describe)


# In[19]:

# get_corr(train_df, sample, 'hotel_roomid')


# ## 历史间隔统计

# In[20]:

span_name, t = '{}_span'.format('hotel_roomid'), 'hotel_roomid'


# In[21]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[22]:

# get_corr(train_df, sa, 'hotel_roomid')


# ## 上次订购的特征 

# In[23]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

