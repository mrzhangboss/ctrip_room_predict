
# coding: utf-8

# In[1]:

import sys
from datetime import datetime
from os.path import join
from warnings import warn
from itertools import chain

import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[2]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[106]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'uid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[4]:

feature_path = join(file_dir, 'user_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[67]:

user_oreder_type = [x for x in train_df.columns if x.startswith('ordertype')]
user_orderbehavior = [x for x in train_df.columns if x.startswith('orderbehavior')]
user_lastord = [x for x in train_df.columns if x.endswith('lastord')]
user_feature_cols = [x for x in train_df.columns if x.startswith('user')]


# ## 添加基本特征

# In[6]:

(now_date - train_df.orderdate_lastord).dt.days.max()


# In[7]:

train_df.orderdate_lastord.min()


# In[37]:

(now_date - train_df.orderdate_lastord).dt.days.isnull().sum()


# In[15]:

sample = extract_feature_count('uid', 'hotel_roomid', train_df, sample)


# In[46]:

def extract_user_feature_is_equal(t, train_df, sample):
    name = '{}_is_equal'.format(t)
    lastord_name = t + '_lastord'
    train_df[name] = np.nan
    train_df.loc[(train_df[lastord_name] == train_df[t]), name] = 1
    sample = extract_feature_count('uid', name, train_df, sample)
    sample = press_date(sample, ['uid' + '__' + name + '_count'])
    return sample


# In[49]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        sample = extract_user_feature_is_equal(t, train_df, sample)


# In[57]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    sample = extract_user_feature_is_equal(t, train_df, sample)


# In[59]:

for c in ['rank', 'star']:
    sample = extract_user_feature_is_equal(c, train_df, sample)


# In[68]:

user_cols = list(chain(user_oreder_type, user_orderbehavior, user_feature_cols))


# In[99]:

add_cols = ['hotel_minprice_lastord', 'basic_minprice_lastord', 'star_lastord'] + user_cols


# In[107]:

not_use = [
    'ordertype_1_ratio', 'ordertype_2_ratio', 'ordertype_4_ratio',
    'ordertype_5_ratio', 'ordertype_7_ratio', 'ordertype_9_ratio',
    'ordertype_11_ratio', 'orderbehavior_1_ratio',
    'orderbehavior_3_ratio_1week', 'orderbehavior_4_ratio_1week',
    'orderbehavior_3_ratio_3month', 'orderbehavior_4_ratio_3month',
    'orderbehavior_5_ratio_3month', 'orderbehavior_6_ratio', 'orderbehavior_8',
    'user_confirmtime', 'user_avgadvanceddate', 'user_avgroomnum',
    'user_avgpromotion', 'user_avgroomarea', 'user_roomservice_4_0ratio',
    'user_roomservice_4_3ratio', 'user_roomservice_4_4ratio',
    'user_roomservice_4_5ratio', 'user_roomservice_3_123ratio',
    'user_roomservice_6_2ratio', 'user_roomservice_6_1ratio',
    'user_roomservice_6_0ratio', 'user_roomservice_5_1ratio',
    'user_roomservice_2_1ratio', 'user_roomservice_8_345ratio',
    'user_ordnum_1week', 'user_roomservice_7_1ratio_1week',
    'user_roomservice_7_0ratio_1week', 'user_roomservice_4_5ratio_1week',
    'user_roomservice_4_3ratio_1week', 'user_roomservice_4_0ratio_1week',
    'user_ordnum_1month', 'user_roomservice_3_123ratio_1month',
    'user_roomservice_7_1ratio_1month', 'user_roomservice_7_0ratio_1month',
    'user_roomservice_4_5ratio_1month', 'user_roomservice_4_3ratio_1month',
    'user_roomservice_4_0ratio_1month', 'user_ordnum_3month',
    'user_roomservice_3_123ratio_3month', 'user_roomservice_7_1ratio_3month',
    'user_roomservice_7_0ratio_3month', 'user_roomservice_4_5ratio_3month',
    'user_roomservice_4_3ratio_3month', 'user_roomservice_4_0ratio_3month',
    'ordertype_3_ratio', 'user_roomservice_4_1ratio'
]
not_use = []


# In[108]:

for col in add_cols:
    if col not in not_use:
        sample = add_column(train_df, sample, 'uid', col)


# In[109]:

get_corr(train_df, sample, 'uid')


# In[74]:

# corr = _

# cor = abs(corr['orderlabel'])

# [x for x in cor.loc[np.isnan(cor)].index]

# [x[4:] for x in cor.loc[cor<0.01].index]  + [x[4:] for x in cor.loc[np.isnan(cor)].index]


# ## 历史价格统计特征 

# In[9]:

name_fmt = '{}_diff_{}'.format('uid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[10]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[ ]:

price_describe = ['mean', '75%']


# In[11]:

sample = extract_value_describe_feature('uid', price_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', hotel_minprice_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', basic_minprice_diff_name, train_df, sample, price_describe)


# In[16]:

# get_corr(train_df, sample, 'uid')


# ### 历史订单间隔统计特征 

# In[38]:

span_name, t = '{}_ordspan'.format('uid'), 'uid'


# In[43]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[42]:

# get_corr(train_df, sample, 'uid')


# In[28]:

train_df.shape


# In[ ]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

