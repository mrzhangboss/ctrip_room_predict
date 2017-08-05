
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
    file_dir = join('..', 'dataset', 'train')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[13]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

sample = pd.read_pickle(join(file_dir, 'uid.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[14]:

feature_path = join(file_dir, 'user_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[15]:

user_oreder_type = [x for x in train_df.columns if x.startswith('ordertype')]
user_orderbehavior = [x for x in train_df.columns if x.startswith('orderbehavior')]
user_lastord = [x for x in train_df.columns if x.endswith('lastord')]
user_feature_cols = [x for x in train_df.columns if x.startswith('user')]


# ## 添加基本特征

# In[16]:

sample = extract_feature_count('uid', 'hotel_roomid', train_df, sample)


# In[17]:

def extract_user_feature_is_equal(t, train_df, sample):
    name = '{}_is_equal'.format(t)
    lastord_name = t + '_lastord'
    train_df[name] = np.nan
    train_df.loc[(train_df[lastord_name] == train_df[t]), name] = 1
    sample = extract_feature_count('uid', name, train_df, sample)
    sample = press_date(sample, ['uid' + '__' + name + '_count'])
    return sample


# In[18]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        sample = extract_user_feature_is_equal(t, train_df, sample)


# In[19]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    sample = extract_user_feature_is_equal(t, train_df, sample)


# In[20]:

for c in ['rank', 'star', 'basicroomid', 'hotelid']: 
    sample = extract_user_feature_is_equal(c, train_df, sample)


# In[21]:

user_cols = list(chain(user_oreder_type, user_orderbehavior, user_feature_cols))


# In[23]:

add_cols = ['hotel_minprice_lastord', 'basic_minprice_lastord', 'star_lastord'] + user_cols


# In[24]:

not_use2 = [
    'user_roomservice_3_123ratio_1month', 'user_roomservice_3_123ratio_1week',
    'user_roomservice_3_123ratio_3month', 'user_avgroomnum',
    'user_avgrecommendlevel', 'user_roomservice_4_3ratio_1week',
    'user_roomservice_4_2ratio', 'user_ordnum_1week',
    'uid_diff_basic_minprice_lastord_mean', 'user_roomservice_4_max_3month',
    'hotelid_is_equal_count', 'user_avggoldstar',
    'orderbehavior_5_ratio_3month', 'roomservice_8_is_equal_count',
    'user_roomservice_4_1ratio_1week', 'user_roomservice_4_max',
    'user_roomservice_8_max', 'user_cvprice', 'user_roomservice_5_1ratio',
    'star_lastord', 'user_roomservice_7_0ratio_3month',
    'orderbehavior_4_ratio_1week', 'user_roomservice_6_2ratio',
    'user_roomservice_8_2ratio', 'user_roomservice_7_0ratio_1month',
    'user_roomservice_8_1ratio', 'user_roomservice_7_0ratio',
    'user_roomservice_5_0ratio', 'user_roomservice_4_max_1month',
    'user_roomservice_4_max_1week', 'user_roomservice_4_2ratio_3month',
    'user_roomservice_4_3ratio_1month', 'user_roomservice_6_0ratio',
    'roomtag_3_is_equal_count', 'user_roomservice_3_123ratio',
    'orderbehavior_3_ratio_1week', 'orderbehavior_7_ratio', 'user_avgstar',
    'user_roomservice_4_4ratio', 'user_roomservice_7_1ratio',
    'user_roomservice_7_max', 'user_roomservice_5_max',
    'user_roomservice_3_max', 'user_roomservice_2_max',
    'roomservice_3_is_equal_count', 'user_roomservice_3_0ratio',
    'user_roomservice_2_0ratio', 'user_roomservice_4_1ratio_1month',
    'user_roomservice_4_1ratio_3month', 'roomservice_2_is_equal_count',
    'user_roomservice_4_0ratio_3month', 'roomservice_5_is_equal_count',
    'user_roomservice_6_1ratio', 'user_roomservice_4_3ratio_3month',
    'user_roomservice_4_4ratio_1week', 'user_roomservice_4_5ratio_1week',
    'user_roomservice_7_1ratio_1week', 'user_avgroomarea',
    'user_roomservice_4_0ratio', 'user_roomservice_8_345ratio',
    'user_roomservice_4_3ratio', 'user_roomservice_2_1ratio',
    'user_roomservice_4_1ratio', 'user_roomservice_4_5ratio',
    'user_roomservice_4_0ratio_1week', 'roomtag_2_is_equal_count',
    'roomid_is_equal_count', 'user_roomservice_4_4ratio_3month',
    'user_roomservice_4_5ratio_3month', 'roomtag_4_is_equal_count',
    'rank_is_equal_count', 'star_is_equal_count', 'orderbehavior_1_ratio',
    'user_roomservice_4_0ratio_1month', 'orderbehavior_3_ratio_3month',
    'orderbehavior_4_ratio_3month', 'user_roomservice_4_4ratio_1month',
    'user_roomservice_4_5ratio_1month', 'orderbehavior_6_ratio'
]
not_use = add_cols


# In[25]:

for col in add_cols:
    if col not in not_use:
        sample = add_column(train_df, sample, 'uid', col)


# In[27]:

# get_corr(train_df, sample, 'uid')


# ## 基本交叉特征

# In[32]:

press_columns = ['uid_user_roomservice_8_2ratio', 'uid_user_roomservice_4_1ratio_3month',
   'uid_user_roomservice_4_1ratio_1month', 'uid_user_roomservice_4_1ratio_1week',
                'uid_user_roomservice_2_0ratio', 'uid_user_roomservice_3_0ratio',
                'uid_user_roomservice_5_0ratio', 'uid_user_roomservice_7_1ratio',
                'uid_user_roomservice_2_max', 'uid_user_roomservice_3_max',
                'uid_user_roomservice_5_max', 'uid_user_roomservice_7_max',
                'uid_user_roomservice_4_max', 'uid_user_roomservice_6_max',
                'uid_user_roomservice_8_max', 'uid_user_roomservice_4_max_1week', 
                'uid_user_roomservice_4_max_1month',
                'uid_user_roomservice_4_max_3month',
                ]


# In[34]:

[x[4:] for x in press_columns]


# In[35]:

# sample["uid_user_roomservice_4_32_rt"]=sample["uid_user_roomservice_4_3ratio"]/sample["uid_user_roomservice_4_2ratio"]
# sample["uid_user_roomservice_4_43_rt"]=sample["uid_user_roomservice_4_4ratio"]/sample["uid_user_roomservice_4_3ratio"]


# ## 历史价格统计特征 

# In[36]:

name_fmt = '{}_diff_{}'.format('uid', '{}')

price_diff_name = name_fmt.format('price_last_lastord')
hotel_minprice_diff_name = name_fmt.format('hotel_minprice_lastord')
basic_minprice_diff_name = name_fmt.format('basic_minprice_lastord')


# In[37]:

train_df[price_diff_name] = train_df['price_deduct'] - train_df['price_last_lastord']
train_df[hotel_minprice_diff_name] = train_df['price_deduct'] - train_df['hotel_minprice_lastord']
train_df[basic_minprice_diff_name] = train_df['price_deduct'] - train_df['basic_minprice_lastord']


# In[38]:

price_describe = ['mean', 'median']


# In[39]:

sample = extract_value_describe_feature('uid', price_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', hotel_minprice_diff_name, train_df, sample, price_describe)
sample = extract_value_describe_feature('uid', basic_minprice_diff_name, train_df, sample, price_describe)


# In[40]:

# get_corr(train_df, sample, 'uid')


# ## 修改特征

# In[41]:

# for i in [1,2,3,4,5,6,7,8,9,10,11]:
#         sample["order_ordertype_%s_num"%i] = sample["uid_ordertype_%s_ratio"%i] * sample["uid_user_ordernum"]
#         del sample["uid_ordertype_%s_ratio"%i]

# for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio"]:
#         sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordernum"]

# [x for x in sample.columns if x.startswith('uid_orderbehavior')]

#  for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week"]:
#         sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordnum_1week"]

# for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month"]:
#         sample["uid_" + c]= sample["uid_" + c] * sample["uid_user_ordnum_3month"]


# In[ ]:

# sample = press_date(sample, ['uid_' + x for x in [
#     "orderbehavior_1_ratio", "orderbehavior_2_ratio", "orderbehavior_6_ratio",
#     "orderbehavior_7_ratio", "orderbehavior_3_ratio_1week",
#     "orderbehavior_4_ratio_1week", "orderbehavior_5_ratio_1week",
#     "orderbehavior_3_ratio_3month",
#     "orderbehavior_4_ratio_3month", "orderbehavior_5_ratio_3month"
# ]])


# ### 历史订单间隔统计特征 

# In[ ]:

span_name, t = '{}_ordspan'.format('uid'), 'uid'


# In[ ]:

# train_df[span_name] = (now_date - train_df.orderdate_lastord).dt.days

# sample = extract_value_describe_feature(t, span_name, train_df, sample, ['max', 'min', 'mean'])


# In[ ]:

# get_corr(train_df, sample, 'uid')


# In[ ]:

train_df.shape


# In[ ]:

for c in not_use2:
    c1 = 'uid__' + c
    c2 = 'uid_' + c
    drop_c = None
    if  c1 in sample.columns:
        drop_c = c1
    elif c2 in sample.columns:
        drop_c = c2
    if drop_c:
        print('drop ', drop_c)
        sample.drop(drop_c, axis=1, inplace=True)


# In[ ]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

