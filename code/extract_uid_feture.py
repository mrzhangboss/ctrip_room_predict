
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


# In[3]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'uid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[4]:

feature_path = join(file_dir, 'user_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[5]:

user_oreder_type = [x for x in train_df.columns if x.startswith('ordertype')]
user_orderbehavior = [x for x in train_df.columns if x.startswith('orderbehavior')]
user_lastord = [x for x in train_df.columns if x.endswith('lastord')]
user_feature_cols = [x for x in train_df.columns if x.startswith('user')]


# ## 添加基本特征

# In[6]:

sample = extract_feature_count('uid', 'hotel_roomid', train_df, sample)


# In[7]:

def extract_user_feature_is_equal(t, train_df, sample):
    name = '{}_is_equal'.format(t)
    lastord_name = t + '_lastord'
    train_df[name] = np.nan
    train_df.loc[(train_df[lastord_name] == train_df[t]), name] = 1
    sample = extract_feature_count('uid', name, train_df, sample)
    sample = press_date(sample, ['uid' + '__' + name + '_count'])
    return sample


# In[8]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        sample = extract_user_feature_is_equal(t, train_df, sample)


# In[9]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    sample = extract_user_feature_is_equal(t, train_df, sample)


# In[10]:

for c in ['rank', 'star', 'basicroomid', 'roomid', 'hotelid']:
    sample = extract_user_feature_is_equal(c, train_df, sample)


# In[11]:

user_cols = list(chain(user_oreder_type, user_orderbehavior, user_feature_cols))


# In[12]:

add_cols = ['hotel_minprice_lastord', 'basic_minprice_lastord', 'star_lastord'] + user_cols


# In[13]:

not_use = []


# In[14]:

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

# In[17]:

press_columns = ['uid_user_roomservice_8_2ratio', 'uid_user_roomservice_4_1ratio_3month',
   'uid_user_roomservice_4_1ratio_1month', 'uid_user_roomservice_4_1ratio_1week',
                'uid_user_roomservice_2_0ratio', 'uid_user_roomservice_3_0ratio',
                'uid_user_roomservice_5_0ratio', 'uid_user_roomservice_7_1ratio',
                'uid_user_roomservice_2_max', 'uid_user_roomservice_3_max',
                'uid_user_roomservice_5_max', 'uid_user_roomservice_7_max',
                'uid_user_roomservice_4_max', 'vuser_roomservice_6_max',
                'uid_user_roomservice_8_max', 'uid_user_roomservice_4_max_1week', 
                'uid_user_roomservice_4_max_1month',
                'uid_user_roomservice_4_max_3month',
                ]


# In[18]:

# sample["uid_user_roomservice_4_32_rt"]=sample["uid_user_roomservice_4_3ratio"]/sample["uid_user_roomservice_4_2ratio"]
# sample["uid_user_roomservice_4_43_rt"]=sample["uid_user_roomservice_4_4ratio"]/sample["uid_user_roomservice_4_3ratio"]


# In[19]:

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


# In[20]:

sample = press_date(sample, press_columns)


# ## 历史价格统计特征 

# In[21]:

name_fmt = '{}_diff_{}'.format('uid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[22]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[23]:

price_describe = ['mean', 'median']


# In[24]:

sample = extract_value_describe_feature('uid', price_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', hotel_minprice_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', basic_minprice_diff_name, train_df, sample, price_describe)


# In[25]:

# get_corr(train_df, sample, 'uid')


# ## 修改特征

# In[26]:

for i in [1,2,3,4,5,6,7,8,9,10,11]:
        sample["order_ordertype_%s_num"%i] = sample["uid_ordertype_%s_ratio"%i] * sample["uid_user_ordernum"]
        del sample["uid_ordertype_%s_ratio"%i]


# In[27]:

for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio"]:
        sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordernum"]


# In[28]:

[x for x in sample.columns if x.startswith('uid_orderbehavior')]


# In[29]:

for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week"]:
       sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordnum_1week"]


# In[30]:

for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month"]:
        sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordnum_3month"]


# In[31]:

sample = press_date(sample, ['uid_' + x for x in [
    "orderbehavior_1_ratio", "orderbehavior_2_ratio", "orderbehavior_6_ratio",
    "orderbehavior_7_ratio", "orderbehavior_3_ratio_1week",
    "orderbehavior_4_ratio_1week", "orderbehavior_5_ratio_1week",
    "orderbehavior_3_ratio_3month",
    "orderbehavior_4_ratio_3month", "orderbehavior_5_ratio_3month"
]])


# ### 历史订单间隔统计特征 

# In[32]:

span_name, t = '{}_ordspan'.format('uid'), 'uid'


# In[33]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[34]:

# get_corr(train_df, sample, 'uid')


# In[35]:

train_df.shape


# In[36]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

