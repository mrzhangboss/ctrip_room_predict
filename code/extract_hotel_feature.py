
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


# In[3]:

# sys.argv[1] = 'test'


# In[4]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[15]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'hotelid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[16]:

feature_path = join(file_dir, 'hotel_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ## 添加基本特征

# In[17]:

sample = add_column(train_df, sample, 'hotelid', 'star')


# ## 上下级关联统计特征

# In[18]:

for f in ['basicroomid', 'roomid']:
    print(datetime.now(), 'begin hotel', f, 'count')
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[19]:

for i in range(8):
    f = 'roomservice_%d' % (i+1)
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[20]:

sample.columns


# In[1]:

# get_corr(train_df, sample[['hotelid', 'hotelid__roomservice_8_count']], 'hotelid')


# In[24]:

pd.read_csv('models/02-0753-importance.txt')


# In[9]:

for i in range(1, 4):
    f = 'roomtag_%d' % (i+1)
    sample = extract_feature_count('hotelid', f, train_df, sample)


# In[10]:

# get_corr(train_df, sample, 'hotelid')


# ### 删除无历史记录却有历史返现值的记录值(默认值为200）

# In[11]:

lastord_cols = [x for x in train_df.columns if x.endswith('lastord')]


# In[12]:

train_df.loc[train_df.orderdate_lastord.isnull(), 'return_lastord'] = np.nan


# In[13]:

train_df.price_deduct.describe()


# ## 基本特征 

# ## 显示的最终价格和原价格的特征 

# In[15]:

use_describe = ['max', 'min', 'median', 'mean', 'std', 'nunique']


# In[16]:

train_df['price_real'] = train_df['price_deduct'] + train_df['returnvalue']


# In[17]:

sample = extract_value_describe_feature('hotelid', 'price_deduct', train_df, sample, use_describe)

sample = extract_value_describe_feature('hotelid', 'price_real', train_df, sample, ['max', 'mean', 'std'])

sample = extract_value_describe_feature('hotelid', 'returnvalue', train_df, sample, ['max', 'mean', 'median'])


# ### 房间的面积统计特征

# ###  删掉为负的值

# In[18]:

train_df.loc[train_df.basic_minarea<0, 'basic_minarea'] = np.nan


# In[19]:

sample = extract_value_describe_feature('hotelid', 'basic_minarea', train_df, sample, ['max', 'mean', 'median'])

sample = extract_value_describe_feature('hotelid', 'basic_maxarea', train_df, sample, ['min', 'mean', 'median'])


# In[20]:

# get_corr(train_df, sample, 'hotelid').tail(25)


# ## 过去物理房型和子房型的统计特征 

# In[21]:

# get_corr(train_df, sample, 'hotelid')


# In[22]:

basic_cols = [
    'basic_week_ordernum_ratio', 'basic_recent3_ordernum_ratio',
    'basic_comment_ratio', 'basic_30days_ordnumratio', 'basic_30days_realratio'
]


# In[23]:

stat_describe = ['min', 'mean', 'max']


# In[24]:

for col in basic_cols:
    sample = extract_value_describe_feature('hotelid', col, train_df, sample, stat_describe)


# In[ ]:

sample = extract_value_describe_feature('hotelid', 'basic_week_ordernum_ratio', train_df, sample, ['count'])


# In[25]:

room_cols = ['room_30days_ordnumratio', 'room_30days_realratio']


# In[26]:

for col in room_cols:
    sample = extract_value_describe_feature('hotelid', col, train_df, sample, ['max', 'min', 'mean'])


# In[ ]:

sample = extract_value_describe_feature('hotelid', 'room_30days_ordnumratio', train_df, sample, ['count'])


# In[27]:

name_fmt = '{}_diff_{}'.format('hotelid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[28]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[29]:

price_desr = ['mean', 'max', 'min']


# In[30]:

sample = extract_value_describe_feature('hotelid', price_diff_name, train_df, sample, price_desr)
sample = extract_value_describe_feature('hotelid', hotel_minprice_diff_name, train_df, sample, price_desr)
sample = extract_value_describe_feature('hotelid', basic_minprice_diff_name, train_df, sample, price_desr)


# In[31]:

# get_corr(train_df, sample, 'hotelid').tail(24)


# ## 历史价格与现在差价统计特征

# In[32]:

hotel_lastord = train_df[[
    'uid', 'hotelid_lastord', 'hotelid', 'star_lastord',
    'hotel_minprice_lastord', 'orderdate_lastord'
]].drop_duplicates()


# In[33]:

if uid_shape != hotel_lastord.shape[0]:
    warn('uid_shape not equal [uid ,hotelid_lastord, hotelid]')


# ## 历史购买时间间隔统计特征 

# In[34]:

span_name, t = '{}_span'.format('hotelid'), 'hotelid'


# In[35]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[36]:

# get_corr(train_df, sample, 'hotelid').tail(8)


# ## 用户过去是否购买或者没有记录

# In[37]:

sample = extract_lastord_is_nan(hotel_lastord, sample, 'hotelid', 'hotelid_lastord')


# In[38]:

# sample = extract_is_lastord(hotel_lastord, sample, 'hotelid', 'hotelid_lastord')


# In[39]:

# get_corr(train_df, sample, 'hotelid').tail()


# In[40]:

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


# In[41]:

# sample = extract_lastord_feature_max_min('star_lastord', hotel_lastord, sample)

# sample = extract_lastord_feature_max_min('hotel_minprice_lastord', hotel_lastord, sample)

# hotel_lastord['orderdate_lastord_days'] = (now_date - hotel_lastord.orderdate_lastord).dt.days

# sample = extract_lastord_feature_max_min('orderdate_lastord_days', hotel_lastord, sample)


# In[42]:

# get_corr(train_df, sample, 'hotelid')


# In[43]:

# sample.info()


# In[44]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

