
# coding: utf-8

# In[35]:

import sys
from datetime import datetime
from os.path import join
from warnings import warn
from itertools import chain

import numpy as np
import pandas as pd
import scipy as sp

from utils import *


# In[36]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[37]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'uid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[38]:

feature_path = join(file_dir, 'user_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[39]:

user_oreder_type = [x for x in train_df.columns if x.startswith('ordertype')]
user_orderbehavior = [x for x in train_df.columns if x.startswith('orderbehavior')]
user_lastord = [x for x in train_df.columns if x.endswith('lastord')]
user_feature_cols = [x for x in train_df.columns if x.startswith('user')]


# ## 添加基本特征

# In[40]:

sample = extract_feature_count('uid', 'hotel_roomid', train_df, sample)


# In[41]:

def extract_user_feature_is_equal(t, train_df, sample):
    name = '{}_is_equal'.format(t)
    lastord_name = t + '_lastord'
    train_df[name] = np.nan
    train_df.loc[(train_df[lastord_name] == train_df[t]), name] = 1
    sample = extract_feature_count('uid', name, train_df, sample)
    sample = press_date(sample, ['uid' + '__' + name + '_count'])
    return sample


# In[42]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        sample = extract_user_feature_is_equal(t, train_df, sample)


# In[43]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    sample = extract_user_feature_is_equal(t, train_df, sample)


# In[44]:

for c in ['rank', 'star']:
    sample = extract_user_feature_is_equal(c, train_df, sample)


# In[45]:

user_cols = list(chain(user_oreder_type, user_orderbehavior, user_feature_cols))


# In[46]:

add_cols = ['hotel_minprice_lastord', 'basic_minprice_lastord', 'star_lastord'] + user_cols


# In[50]:

not_use = []


# In[51]:

for col in add_cols:
    if col not in not_use:
        sample = add_column(train_df, sample, 'uid', col)


# In[15]:

# get_corr(train_df, sample, 'uid')


# In[16]:

# corr = _

# cor = abs(corr['orderlabel'])

# [x for x in cor.loc[np.isnan(cor)].index]

# [x[4:] for x in cor.loc[cor<0.01].index]  + [x[4:] for x in cor.loc[np.isnan(cor)].index]


# ## 基本交叉特征

# In[56]:

sample.columns


# In[66]:

press_columns = ['uid_user_roomservice_8_2ratio', 'uid_user_roomservice_4_1ratio_3month',
   'uid_user_roomservice_4_1ratio_1month', 'uid_user_roomservice_4_1ratio_1week',
                'uid_user_roomservice_2_0ratio', 'uid_user_roomservice_3_0ratio',
                'uid_user_roomservice_5_0ratio', 'uid_user_roomservice_7_1ratio',
                'uid_user_roomservice_2_max', 'uid_user_roomservice_3_max',
                'uid_user_roomservice_5_max', 'uid_user_roomservice_7_max',
                'uid_user_roomservice_4_max', 'vuser_roomservice_6_max',
                'uid_user_roomservice_8_max', 'uid_user_roomservice_4_max_1week', 
                'uid_user_roomservice_4_max_1month',
                'uid_user_roomservice_4_max_3month']


# In[67]:

# sample["user_roomservice_8_345ratio"]=sample["user_roomservice_5_345ratio"]
# del sample["user_roomservice_5_345ratio"]
sample[
    "uid_user_roomservice_8_2ratio"] = 1 - sample["uid_user_roomservice_8_345ratio"] - sample["uid_user_roomservice_8_1ratio"]
sample[
    "uid_user_roomservice_4_1ratio_3month"] = 1 - sample["uid_user_roomservice_4_0ratio_3month"] - sample["uid_user_roomservice_4_2ratio_3month"] - sample["uid_user_roomservice_4_3ratio_3month"] - sample["uid_user_roomservice_4_4ratio_3month"] - sample["uid_user_roomservice_4_5ratio_3month"]
sample[
    "uid_user_roomservice_4_1ratio_1month"] = 1 - sample["uid_user_roomservice_4_0ratio_1month"] - sample["uid_user_roomservice_4_2ratio_1month"] - sample["uid_user_roomservice_4_3ratio_1month"] - sample["uid_user_roomservice_4_4ratio_1month"] - sample["uid_user_roomservice_4_5ratio_1month"]
sample[
    "uid_user_roomservice_4_1ratio_1week"] = 1 - sample["uid_user_roomservice_4_0ratio_1week"] - sample["uid_user_roomservice_4_2ratio_1week"] - sample["uid_user_roomservice_4_3ratio_1week"] - sample["uid_user_roomservice_4_4ratio_1week"] - sample["uid_user_roomservice_4_5ratio_1week"]
sample["uid_user_roomservice_2_0ratio"] = 1 - sample["uid_user_roomservice_2_1ratio"]
sample["uid_user_roomservice_3_0ratio"] = 1 - sample["uid_user_roomservice_3_123ratio"]
sample["uid_user_roomservice_5_0ratio"] = 1 - sample["uid_user_roomservice_5_1ratio"]
sample["uid_user_roomservice_7_1ratio"] = 1 - sample["uid_user_roomservice_7_0ratio"]
sample["uid_user_roomservice_2_max"] = np.argmax(
    sample[["uid_user_roomservice_2_%sratio" % i for i in range(2)]].values,
    axis=1)
sample["uid_user_roomservice_3_max"] = np.argmax(
    sample[["uid_user_roomservice_3_%sratio" % i for i in [0, 123]]].values,
    axis=1)
sample["uid_user_roomservice_5_max"] = np.argmax(
    sample[["uid_user_roomservice_5_%sratio" % i for i in range(2)]].values,
    axis=1)
sample["uid_user_roomservice_7_max"] = np.argmax(
    sample[["uid_user_roomservice_7_%sratio" % i for i in range(2)]].values,
    axis=1)
sample["uid_user_roomservice_4_max"] = np.argmax(
    sample[["uid_user_roomservice_4_%sratio" % i for i in range(6)]].values,
    axis=1)
sample["vuser_roomservice_6_max"] = np.argmax(
    sample[["uid_user_roomservice_6_%sratio" % i for i in range(3)]].values,
    axis=1)
sample["uid_user_roomservice_8_max"] = np.argmax(
    sample[["uid_user_roomservice_8_%sratio" % i for i in [1, 2, 345]]].values,
    axis=1)
sample["uid_user_roomservice_4_max_1week"] = np.argmax(
    sample[["uid_user_roomservice_4_%sratio_1month" % i for i in range(6)]].values,
    axis=1)
sample["uid_user_roomservice_4_max_1month"] = np.argmax(
    sample[["uid_user_roomservice_4_%sratio_1month" % i for i in range(6)]].values,
    axis=1)
sample["uid_user_roomservice_4_max_3month"] = np.argmax(
    sample[["uid_user_roomservice_4_%sratio_3month" % i for i in range(6)]].values,
    axis=1)


# In[68]:

sample = press_date(sample, press_columns)


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

price_describe = ['mean', 'median']


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

