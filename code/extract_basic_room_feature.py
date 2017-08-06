
# coding: utf-8

# In[1]:

import sys
import gc
from datetime import datetime
from os.path import join
from warnings import warn

import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[2]:

# sys.argv[1] = 'test'


# In[3]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    is_test = False
    file_dir = join('..', 'dataset', 'train')
else:
    is_test = True
    file_dir = join('..', 'dataset',  dir_arg)


# In[4]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'basicroomid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[5]:

feature_path = join(file_dir, 'basic_room_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ## 基本分类计数特征

# In[6]:

train_df.loc[train_df.basic_minarea<0, 'basic_minarea'] = np.nan
train_df.loc[train_df.basic_maxarea<0, 'basic_maxarea'] = np.nan


# In[7]:

sample = add_column(train_df, sample, 'basicroomid', 'basic_minarea')
sample = add_column(train_df, sample, 'basicroomid', 'basic_maxarea')


# In[8]:

basic_cols = [
    'basic_week_ordernum_ratio', 'basic_recent3_ordernum_ratio',
    'basic_comment_ratio', 'basic_30days_ordnumratio', 'basic_30days_realratio'
]


# In[10]:

# for col in basic_cols:
#     sample = add_column(train_df, sample, 'basicroomid', col)


# In[11]:

for i in range(1, 8):
    f = 'roomservice_%d' % (i+1)
    sample = extract_feature_count('basicroomid', f, train_df, sample)


# In[12]:

for i in range(4):
    f = 'roomtag_%d' % (i+1)
    sample = extract_feature_count('basicroomid', f, train_df, sample)


# In[13]:

sample = extract_feature_count('basicroomid', 'roomid', train_df, sample)


# In[14]:

# get_corr(train_df, sample, 'basicroomid')


# ## 数值统计特征

# ### 价格

# In[15]:

use_describe = ['max', 'min', 'median', 'mean', 'std', 'nunique']


# In[16]:

train_df['price_real'] = train_df['price_deduct'] + train_df['returnvalue']


# In[17]:

sample = extract_value_describe_feature('basicroomid', 'price_deduct', train_df, sample, use_describe)

sample = extract_value_describe_feature('basicroomid', 'price_real', train_df, sample, ['max', 'mean', 'min', 'median'])

sample = extract_value_describe_feature('basicroomid', 'returnvalue', train_df, sample,['max', 'min', 'median'] )


# ### 价格排序

# In[18]:

def df_min_orderid(df):
    add = pd.DataFrame(df.groupby(["orderid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "orderid_price_deduct_min"]
    df = df.merge(add, on=["orderid"], how="left")
    df = press_date(df, ['orderid_price_deduct_min'])
    return df


# In[19]:

def df_rank_mean(df):
    add = pd.DataFrame(df.groupby(["basicroomid"]).orderid_price_deduct_min_rank.mean()).reset_index()
    add.columns = ["basicroomid","orderid_price_deduct_min_rank_mean"]
    df = df.merge(add, on=["basicroomid"], how="left")
    df = press_date(df, ['orderid_price_deduct_min_rank_mean'])
    return df


# In[20]:

# train_df = df_min_orderid(df)

# train_df["orderid_price_deduct_min_rank"] = train_df['orderid_price_deduct_min'].groupby(train_df['orderid']).rank()

# train_df["orderid_price_deduct_min_rank"]

# train_df = df_rank_mean(train_df)


# In[21]:

# sample['basicroomid__price_deduct_min_rank'] = sample.basicroomid__price_deduct_min.rank()


# ## 子房型rank统计特征

# In[22]:

sample = extract_value_describe_feature('basicroomid', 'rank',
                                        train_df, sample,
                                        ['max', 'min', 'median', 'mean', 'std', 'nunique'])


# In[23]:

# get_corr(train_df, sample, 'basicroomid')


# ## 物理房型统计特征 

# In[19]:

basic_stat_cols = [
    'basicroomid', 'orderdate', 'basic_week_ordernum_ratio',
    'basic_recent3_ordernum_ratio', 'basic_comment_ratio',
    'basic_30days_ordnumratio', 'basic_30days_realratio'
]


# In[59]:

if is_test:
    history_df = pd.read_pickle('../dataset/train.pkl')
    history_df = history_df.loc[history_df.basicroomid.isin(train_df.basicroomid.unique()), basic_stat_cols]
    print('get test_df ', history_df.shape)
    tdf = train_df[basic_stat_cols]
    tdf = pd.concat([tdf, history_df])
    del history_df
    gc.collect()
else:
    tdf = train_df[basic_stat_cols]


# In[61]:

tdf.orderdate = tdf.orderdate.dt.weekday

ntdf = tdf[basic_stat_cols].groupby(['basicroomid', 'orderdate']).mean().reset_index()


# In[62]:

stat_cols = [
    'basic_week_ordernum_ratio', 'basic_recent3_ordernum_ratio',
    'basic_comment_ratio', 'basic_30days_ordnumratio', 'basic_30days_realratio'
]


# In[64]:

use_describe = ['max', 'mean', 'mad', 'var', 'median', 'sum']


# In[51]:

# sample = extract_value_describe_feature('basicroomid', 'basic_week_ordernum_ratio_var', ntdf, sample,
#                                         ['max', 'mean', 'median', 'sum'])


# In[ ]:

['basic_week_ordernum_ratio_mad', 'basicroomid__basic_week_ordernum_ratio_var',
'basicroomid__basic_recent3_ordernum_ratio_mad', 
'basicroomid__basic_recent3_ordernum_ratio_var',
'basicroomid__basic_comment_ratio_var',
'basicroomid__basic_30days_ordnumratio_mad',
'basicroomid__basic_30days_ordnumratio_var',
'basicroomid__basic_30days_realratio_var']


# In[65]:

for c in stat_cols:
    print('extract',  c, use_describe)
    sample = extract_value_describe_feature(
        'basicroomid', c, ntdf, sample, use_describe)


# In[67]:

# get_corr(train_df, sample, 'basicroomid')


# ## 子房型的统计特征 

# In[24]:

room_cols = ['room_30days_ordnumratio', 'room_30days_realratio']


# In[25]:

sample = extract_value_describe_feature(
    'basicroomid', 'room_30days_ordnumratio', train_df, sample,
    ['max', 'min', 'median', 'mean', 'std', 'nunique', 'var', 'mad', 'sum'])


# In[26]:

sample = extract_value_describe_feature('basicroomid', 'room_30days_realratio',
                                        train_df, sample,
                                        ['max', 'min', 'median', 'mean', 'std', 'nunique', 'count',  'var', 'mad', 'sum'])


# In[27]:

# get_corr(train_df, sample, 'basicroomid').tail(10)


# ## 历史价格统计特征

# In[28]:

price_use_describe = ['max', 'std', 'mean', 'min']


# In[29]:

name_fmt = '{}_diff_{}'.format('basicroomid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[30]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[31]:

sample = extract_value_describe_feature('basicroomid', price_diff_name, train_df, sample, price_use_describe)


# In[32]:

sample = extract_value_describe_feature('basicroomid', hotel_minprice_diff_name, train_df, sample, price_use_describe)
sample = extract_value_describe_feature('basicroomid', basic_minprice_diff_name, train_df, sample, price_use_describe)


# In[33]:

# get_corr(train_df, sample, 'basicroomid').tail(20)


# ## 历史时间间隔统计特征

# In[34]:

span_name, t = '{}_span'.format('basicroomid'), 'basicroomid'


# In[35]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[36]:

# get_corr(train_df, sample, 'basicroomid')


# ## 上次订购的特征 

# In[37]:

# basic_cols = [c for c in train_df.columns if c.startswith('basic') and not c.endswith('area')]
# basic_cols


# In[79]:

# use_cols = ['uid', 'orderdate_lastord', 'hotelid', 'basicroomid_lastord', 'basicroomid'] 


# In[80]:

# basic_feature = train_df[use_cols].drop_duplicates()


# In[81]:

# if train_df.drop_duplicates(['uid', 'basicroomid']).shape[0] != basic_feature.shape[0]:
#     warn('[uid, basicroomid].shape[0] != basic_feature.shape[0]')


# In[82]:

# cols = [x for x in train_df.columns if x.endswith('lastord')]


# In[88]:

# train_df.loc[train_df.orderdate_lastord.isnull()][cols].return_lastord.value_counts()


# In[89]:

# sample = extract_lastord_is_nan(basic_feature, sample, 'basicroomid', 'basicroomid_lastord')


# In[85]:

# sample = extract_is_lastord(basic_feature, sample, 'basicroomid', 'basicroomid_lastord')


# In[38]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

