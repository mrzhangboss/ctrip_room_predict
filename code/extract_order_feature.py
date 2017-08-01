
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

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[4]:

feature_path = join(file_dir, 'order_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# ## 基础特征

# #### 排序特征

# In[5]:

def df_roomrank_mean(df):
    add = pd.DataFrame(df.groupby(["roomid"]).order_basicroomid_price_rank.mean()).reset_index()
    add.columns = ["roomid","order_basicroomid_price_rank_mean"]
    df = df.merge(add, on=["roomid"], how="left")
    df = press_date(df, ['order_basicroomid_price_rank_mean'])
    return df


# In[6]:

train_df['order_basicroomid_price_rank'] = train_df['price_deduct'].groupby([train_df['orderid'], train_df['basicroomid']]).rank()


# In[7]:

train_df = df_roomrank_mean(train_df)


# In[8]:

# train_df = press_date(train_df, ['order_basicroomid_price_rank'])


# In[9]:

# 每个basicid价格的中位数
def df_median(df):
    add = pd.DataFrame(df.groupby(["orderid", "basicroomid"]).price_deduct.median()).reset_index()
    add.columns = ["orderid", "basicroomid", "basicroomid_price_deduct_median"]
    df = df.merge(add, on=["orderid", "basicroomid"], how="left")
    return df

# 每个basicid价格的最小值
def df_min(df):
    add = pd.DataFrame(df.groupby(["orderid", "basicroomid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "basicroomid", "basicroomid_price_deduct_min"]
    df = df.merge(add, on=["orderid", "basicroomid"], how="left")
    return df

# 每个orderid价格的最小值
def df_min_orderid(df):
    add = pd.DataFrame(df.groupby(["orderid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "orderid_price_deduct_min"]
    df = df.merge(add, on=["orderid"], how="left")
    return df

#排序特征
def df_rank_mean(df):
    add = pd.DataFrame(df.groupby(["basicroomid"]).orderid_price_deduct_min_rank.mean()).reset_index()
    add.columns = ["basicroomid","orderid_price_deduct_min_rank_mean"]
    df = df.merge(add, on=["basicroomid"], how="left")
    return df

def df_roomrank_mean(df):
    add = pd.DataFrame(df.groupby(["roomid"]).basicroomid_price_rank.mean()).reset_index()
    add.columns = ["roomid","basicroomid_price_rank_mean"]
    df = df.merge(add, on=["roomid"], how="left")
    return df


# In[10]:

def df_min(df):
    add = pd.DataFrame(df.groupby(["orderid", "basicroomid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "basicroomid", "basicroomid_price_deduct_min"]
    df = df.merge(add, on=["orderid", "basicroomid"], how="left")
#     df = press_date(df, ['basicroomid_price_deduct_min'])
    return df


# ### 上次订购的价格和当时最低价的比

# In[11]:

train_df=df_median(train_df)
train_df=df_min(train_df)
train_df=df_min_orderid(train_df)


# In[12]:

for x in train_df.columns:
    if type(train_df[x]) != pd.Series:
        print(x, type(train_df[x]))


# In[13]:

press_columns = [
    'order_basic_minprice_rt', 'price_dif', 'price_dif_hotel_hotel',
    'price_dif_basic_hotel', 'price_dif_basic_hotel_rt', 'price_dif_basic',
    'return_dx', 'price_tail1', 'rank_equal', 'area_price',
    'price_dif_basic_rt', 'price_ori',
    'basicroomid_price_deduct_min_minprice_rt', 'price_max_min_rt',
    'price_dif_rt', 'price_dx', 'city_num', 'price_dif_hotel_hotel_rt',
    'price_dif_hotel_rt', 'basicroomid_price_deduct_min', 'price_dif_hotel'
]


# In[14]:

train_df["city_num"]=train_df["user_ordernum"]/train_df["user_citynum"]
train_df["area_price"]=train_df["user_avgprice"]/train_df["user_avgroomarea"]
train_df["price_max_min_rt"]=train_df["user_maxprice"]/train_df["user_minprice"]
train_df["basicroomid_price_deduct_min_minprice_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["user_minprice"]

train_df["price_dif"]=train_df["basicroomid_price_deduct_min"]-train_df["price_deduct"]
train_df["price_dif_hotel"]=train_df["basicroomid_price_deduct_min"]-train_df["hotel_minprice_lastord"]
train_df["price_dif_basic"]=train_df["basicroomid_price_deduct_min"]-train_df["basic_minprice_lastord"]

train_df["price_dif_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["price_deduct"]
train_df["price_dif_hotel_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["hotel_minprice_lastord"]
train_df["price_dif_basic_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["basic_minprice_lastord"]

train_df["price_dif_hotel"]=train_df["orderid_price_deduct_min"]-train_df["price_deduct"]
train_df["price_dif_hotel_hotel"]=train_df["orderid_price_deduct_min"]-train_df["hotel_minprice_lastord"]
train_df["price_dif_basic_hotel"]=train_df["orderid_price_deduct_min"]-train_df["basic_minprice_lastord"]

train_df["price_dif_hotel_rt"]=train_df["orderid_price_deduct_min"]/train_df["price_deduct"]
train_df["price_dif_hotel_hotel_rt"]=train_df["orderid_price_deduct_min"]/train_df["hotel_minprice_lastord"]
train_df["price_dif_basic_hotel_rt"]=train_df["orderid_price_deduct_min"]/train_df["basic_minprice_lastord"]

#train_df["order_basic_minprice_dif"]=train_df["basicroomid_price_deduct_min"]-train_df["orderid_price_deduct_min"]
train_df["order_basic_minprice_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["orderid_price_deduct_min"]
#train_df["hotel_basic_minprice_lastord_rt"]=train_df["basic_minprice_lastord"]/train_df["hotel_minprice_lastord"]



train_df["price_tail1"]=train_df["price_deduct"]%10
train_df.loc[(train_df.price_tail1==4)|(train_df.price_tail1==7), "price_tail1"]= 1
train_df.loc[(train_df.price_tail1!=4)&(train_df.price_tail1!=7), "price_tail1"]= 0


#del train_df["hotelid_lastord"]
train_df["rank_equal"]= (train_df["rank"] ==train_df["rank_lastord"]).astype(np.int8)

#价格高低
train_df["price_dx"] = train_df["price_deduct"] - train_df["price_last_lastord"] 

train_df["return_dx"] = train_df["returnvalue"] - train_df["return_lastord"]

train_df["price_ori"] = train_df["price_deduct"] + train_df["returnvalue"]


# In[15]:

train_df["order_hotel_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["basic_minprice_lastord"]
train_df["order_hotel_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["basic_minprice_lastord"]


# In[16]:

train_df = press_date(train_df, ['order_hotel_last_price_min_rt', 'order_basic_last_price_min_rt', 'order_hotel_last_price_min_dif', 'order_basic_last_price_min_dif'])


# In[17]:

train_df['orderspan'] = (now_date - train_df['orderdate_lastord']).dt.days.astype(np.float16)


# In[18]:

train_df['orderhour'] = train_df['orderdate'].dt.hour.astype(np.int8)


# In[19]:

for c in train_df.columns:
    if train_df[c].dtype == np.object:
        print(c)


# In[20]:

train_df = press_date(train_df, press_columns)


# In[21]:

order_cols = [
    'orderhour', 'orderid', 'uid', 'hotelid', 'basicroomid', 'hotel_roomid',
    'roomid', 'orderlabel', 'orderspan', 'order_basicroomid_price_rank',
    'order_basicroomid_price_rank_mean', 'order_hotel_last_price_min_rt',
    'order_basic_last_price_min_rt', 'order_hotel_last_price_min_dif',
    'order_basic_last_price_min_dif'
] + press_columns


# In[22]:

room_service = ['roomservice_%d' % i for i in range(1, 9)]


# In[23]:

room_tag = ['roomtag_%d' % i for i in range(1, 5)]


# In[24]:

total_cols = list(chain(order_cols, room_service, room_tag))


# In[25]:

sample = train_df[total_cols]


# In[26]:

for x in train_df.columns:
    if type(train_df[x]) != pd.Series:
        print(x, type(train_df[x]))


# In[28]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

