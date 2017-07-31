
# coding: utf-8

# In[19]:

import sys
from datetime import datetime
from os.path import join
from warnings import warn


import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[20]:

# sys.argv[1] = 'test'


# In[21]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[22]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'hotelid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[23]:

feature_path = join(file_dir, 'hotel_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ## 添加基本特征

# In[24]:

sample = add_column(train_df, sample, 'hotelid', 'star')


# ## 上下级关联统计特征

# In[25]:

for f in ['basicroomid', 'roomid']:
    print(datetime.now(), 'begin hotel', f, 'count')
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[26]:

for i in range(8):
    f = 'roomservice_%d' % (i+1)
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[27]:

for i in range(1, 4):
    f = 'roomtag_%d' % (i+1)
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[28]:

# get_corr(train_df, sample, 'hotelid')


# ### 删除无历史记录却有历史返现值的记录值(默认值为200）

# In[29]:

lastord_cols = [x for x in train_df.columns if x.endswith('lastord')]


# In[30]:

train_df.loc[train_df.orderdate_lastord.isnull(), 'return_lastord'] = np.nan


# In[31]:

train_df.price_deduct.describe()


# ## 基本特征 

# ## 显示的最终价格和原价格的特征 

# In[32]:

use_describe = ['max', 'min', '75%', 'mean', 'std']


# In[33]:

train_df['price_real'] = train_df['price_deduct'] + train_df['returnvalue']


# In[15]:

sample = extract_value_describe_feature('hotelid', 'price_deduct', train_df, sample, use_describe)

sample = extract_value_describe_feature('hotelid', 'price_real', train_df, sample, ['max', 'mean', 'std'])

sample = extract_value_describe_feature('hotelid', 'returnvalue', train_df, sample, ['max', 'mean', '75%'])


# ### 房间的面积统计特征

# ###  删掉为负的值

# In[5]:

train_df.loc[train_df.basic_minarea<0, 'basic_minarea'] = np.nan


# In[6]:

sample = extract_value_describe_feature('hotelid', 'basic_minarea', train_df, sample, ['max', 'mean', '75%'])

sample = extract_value_describe_feature('hotelid', 'basic_maxarea', train_df, sample, ['min', 'mean', '25%'])


# In[7]:

# get_corr(train_df, sample, 'hotelid').tail(25)


# ## 过去物理房型和子房型的统计特征 

# In[ ]:

# get_corr(train_df, sample, 'hotelid')


# In[8]:

basic_cols = [
    'basic_week_ordernum_ratio', 'basic_recent3_ordernum_ratio',
    'basic_comment_ratio', 'basic_30days_ordnumratio', 'basic_30days_realratio'
]


# In[ ]:

stat_describe = ['min', 'mean']


# In[9]:

for col in basic_cols:
    sample = extract_value_describe_feature('hotelid', col, train_df, sample, stat_describe)


# In[11]:

room_cols = ['room_30days_ordnumratio', 'room_30days_realratio']


# In[12]:

for col in room_cols:
    sample = extract_value_describe_feature('hotelid', col, train_df, sample, ['max', 'min'])


# In[15]:

name_fmt = '{}_diff_{}'.format('hotelid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[16]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[ ]:

price_desr = ['mean', 'max', 'min']


# In[17]:

sample = extract_value_describe_feature('hotelid', price_diff_name, train_df, sample, price_desr)
sample = extract_value_describe_feature('hotelid', hotel_minprice_diff_name, train_df, sample, price_desr)
sample = extract_value_describe_feature('hotelid', basic_minprice_diff_name, train_df, sample, price_desr)


# In[18]:

# get_corr(train_df, sample, 'hotelid').tail(24)


# ## 历史价格与现在差价统计特征

# In[24]:

hotel_lastord = train_df[[
    'uid', 'hotelid_lastord', 'hotelid', 'star_lastord',
    'hotel_minprice_lastord', 'orderdate_lastord'
]].drop_duplicates()


# In[25]:

if uid_shape != hotel_lastord.shape[0]:
    warn('uid_shape not equal [uid ,hotelid_lastord, hotelid]')


# ## 历史购买时间间隔统计特征 

# In[64]:

span_name, t = '{}_span'.format('hotelid'), 'hotelid'


# In[65]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[69]:

# get_corr(train_df, sample, 'hotelid').tail(8)


# ## 用户过去是否购买或者没有记录

# In[26]:

sample = extract_lastord_is_nan(hotel_lastord, sample, 'hotelid', 'hotelid_lastord')


# In[27]:

sample = extract_is_lastord(hotel_lastord, sample, 'hotelid', 'hotelid_lastord')


# In[28]:

# get_corr(train_df, sample, 'hotelid').tail()


# In[29]:

def extract_lastord_feature_max_min(t, hotel_lastord, sample):
    min_fmt = '{}_min'.format(t)
    max_fmt = '{}_max'.format(t)
    series = hotel_lastord.groupby('hotelid')[t].min()
    series.name = min_fmt
    sample = sample.join(series, on='hotelid')
    series = hotel_lastord.groupby('hotelid')[t].max()
    series.name = max_fmt
    sample = sample.join(series, on='hotelid')
    max_equal_min_fmt = '{}_max_equeal_min'.format(t)
    sample[max_equal_min_fmt] = (sample[max_fmt] == sample[min_fmt]).astype(np.int8)
    sample = press_date(sample, [max_equal_min_fmt, max_fmt, min_fmt])
    return sample


# In[30]:

# sample = extract_lastord_feature_max_min('star_lastord', hotel_lastord, sample)

# sample = extract_lastord_feature_max_min('hotel_minprice_lastord', hotel_lastord, sample)

# hotel_lastord['orderdate_lastord_days'] = (now_date - hotel_lastord.orderdate_lastord).dt.days

# sample = extract_lastord_feature_max_min('orderdate_lastord_days', hotel_lastord, sample)


# In[31]:

# get_corr(train_df, sample, 'hotelid')


# In[32]:

# sample.info()


# In[33]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

