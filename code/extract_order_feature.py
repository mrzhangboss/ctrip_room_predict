
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


# In[3]:

train_df = pd.read_pickle(join(file_dir, 'base_feauture.pkl'))

now_date = train_df.orderdate.max().date()
print(datetime.now(), now_date)

uid_shape, hotelid_shape, basicroomid_shape, roomid_shape = print_shape(
    train_df, ['uid', 'hotelid', 'basicroomid', 'roomid'])


# In[4]:

feature_path = join(file_dir, 'order_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[5]:

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


# ## 基础特征

# #### 排序特征

# In[6]:

train_df=df_median(train_df)
train_df=df_min(train_df)
train_df=df_min_orderid(train_df)


# In[7]:

train_df['basicroomid_price_rank'] = train_df['price_deduct'].groupby([train_df['orderid'], train_df['basicroomid']]).rank()


# In[8]:

train_df = df_roomrank_mean(train_df)


# In[9]:

train_df["orderid_price_deduct_min_rank"] = train_df['orderid_price_deduct_min'].groupby(train_df['orderid']).rank()


# In[10]:

train_df = df_rank_mean(train_df)


# In[11]:

# train_df = press_date(train_df, ['order_basicroomid_price_rank'])


# ### 上次订购的价格和当时最低价的比

# In[12]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
        train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[13]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
    train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[14]:

for t in ['rank', 'star', 'basicroomid', 'roomid', 'hotelid']:
    train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
    train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[15]:

train_df['order_weekday'] = train_df.orderdate.dt.weekday


# In[16]:

train_df['order_weekday_lastord'] = train_df.orderdate_lastord.dt.weekday


# In[17]:

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

train_df["order_basic_minprice_rt"]=train_df["basicroomid_price_deduct_min"]/train_df["orderid_price_deduct_min"]



train_df["price_tail1"]=train_df["price_deduct"]%10
# train_df.loc[(train_df.price_tail1==4)|(train_df.price_tail1==7), "price_tail1"]= 1
# train_df.loc[(train_df.price_tail1!=4)&(train_df.price_tail1!=7), "price_tail1"]= 0


train_df["price_dx"] = train_df["price_deduct"] - train_df["price_last_lastord"] 

train_df["return_dx"] = train_df["returnvalue"] - train_df["return_lastord"]

train_df["price_ori"] = train_df["price_deduct"] + train_df["returnvalue"]


# In[18]:

train_df["price_star"]=train_df["price_deduct"]/(train_df["star"])
train_df["price_minarea"]=train_df["price_deduct"]/(train_df["basic_minarea"]-1)

train_df["star_dif"]=train_df["user_avgstar"]-train_df["star"]

train_df["price_ave_dif_rt"]=train_df["price_deduct"]/train_df["user_avgdealprice"]
train_df["price_ave_star_dif"]=train_df["price_deduct"]/train_df["user_avgprice_star"]
train_df["price_h_w_rt"]=train_df["user_avgdealpriceholiday"]/train_df["user_avgdealpriceworkday"]

train_df["price_ave_dif"] = train_df["price_deduct"] - train_df["user_avgdealprice"]


# In[19]:

train_df["order_hotel_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["basic_minprice_lastord"]
train_df["order_hotel_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["basic_minprice_lastord"]


# In[20]:

train_df = press_date(train_df, ['order_hotel_last_price_min_rt', 'order_basic_last_price_min_rt', 'order_hotel_last_price_min_dif', 'order_basic_last_price_min_dif'])


# ### 用户特征 

# In[21]:

train_df["user_roomservice_8_2ratio"]=1-train_df["user_roomservice_8_345ratio"]-train_df["user_roomservice_8_1ratio"]
train_df["user_roomservice_4_1ratio_3month"] = 1 - train_df["user_roomservice_4_0ratio_3month"] - train_df["user_roomservice_4_2ratio_3month"] - train_df["user_roomservice_4_3ratio_3month"] - train_df["user_roomservice_4_4ratio_3month"] - train_df["user_roomservice_4_5ratio_3month"]
train_df["user_roomservice_4_1ratio_1month"] = 1 - train_df["user_roomservice_4_0ratio_1month"] - train_df["user_roomservice_4_2ratio_1month"] - train_df["user_roomservice_4_3ratio_1month"] - train_df["user_roomservice_4_4ratio_1month"] - train_df["user_roomservice_4_5ratio_1month"]
train_df["user_roomservice_4_1ratio_1week"] = 1 - train_df["user_roomservice_4_0ratio_1week"] - train_df["user_roomservice_4_2ratio_1week"] - train_df["user_roomservice_4_3ratio_1week"] - train_df["user_roomservice_4_4ratio_1week"] - train_df["user_roomservice_4_5ratio_1week"]
train_df["user_roomservice_2_0ratio"]=1-train_df["user_roomservice_2_1ratio"]
train_df["user_roomservice_3_0ratio"]=1-train_df["user_roomservice_3_123ratio"]
train_df["user_roomservice_5_0ratio"]=1-train_df["user_roomservice_5_1ratio"]
train_df["user_roomservice_7_1ratio"]=1-train_df["user_roomservice_7_0ratio"]
train_df["user_roomservice_2_max"] = np.argmax(train_df[["user_roomservice_2_%sratio" % i for i in range(2)]].values, axis=1)
train_df["user_roomservice_3_max"] = np.argmax(train_df[["user_roomservice_3_%sratio" % i for i in [0,123]]].values, axis=1)
train_df["user_roomservice_5_max"] = np.argmax(train_df[["user_roomservice_5_%sratio" % i for i in range(2)]].values, axis=1)
train_df["user_roomservice_7_max"] = np.argmax(train_df[["user_roomservice_7_%sratio" % i for i in range(2)]].values, axis=1)
train_df["user_roomservice_4_max"]=np.argmax(train_df[["user_roomservice_4_%sratio"%i for i in range(6)]].values,axis=1)
train_df["user_roomservice_6_max"]=np.argmax(train_df[["user_roomservice_6_%sratio"%i for i in range(3)]].values,axis=1)
train_df["user_roomservice_8_max"]=np.argmax(train_df[["user_roomservice_8_%sratio"%i for i in [1,2,345]]].values,axis=1)
train_df["user_roomservice_4_max_1week"]=np.argmax(train_df[["user_roomservice_4_%sratio_1month"%i for i in range(6)]].values,axis=1)
train_df["user_roomservice_4_max_1month"]=np.argmax(train_df[["user_roomservice_4_%sratio_1month"%i for i in range(6)]].values,axis=1)
train_df["user_roomservice_4_max_3month"]=np.argmax(train_df[["user_roomservice_4_%sratio_3month"%i for i in range(6)]].values,axis=1)
train_df["roomservice_8"]=train_df["roomservice_8"].apply(lambda x:2 if x>2 else x-1)
train_df["roomservice_3"]=train_df["roomservice_3"].apply(lambda x:1 if x>0 else 0)


# In[22]:

for i in [1,2,3,4,5,6,7,8,9,10,11]:
    train_df["order_ordertype_%s_num"%i] = train_df["ordertype_%s_ratio"%i] * train_df["user_ordernum"]
    del train_df["ordertype_%s_ratio"%i]

for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio"]:
    train_df[c]= train_df[c] * train_df["user_ordernum"]

for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week"]:
    train_df[c]= train_df[c] * train_df["user_ordnum_1week"]

for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month"]:
    train_df[c]= train_df[c] * train_df["user_ordnum_3month"]


# In[23]:

train_df['orderspan'] = (now_date - train_df['orderdate_lastord']).dt.days.astype(np.float16)


# In[24]:

train_df['orderhour'] = train_df['orderdate'].dt.hour.astype(np.int8)


# ## 交叉特征

# In[25]:

train_df['rank_roomservice_8'] = (
    train_df['roomservice_8'].astype(str) +
    train_df['rank'].astype(str)).astype('category').cat.codes


# In[26]:

use_cols = [x for x in train_df.columns if x not in ['orderdate', 'orderdate_lastord']]


# In[27]:

train_df = press_date(train_df, [x for x in use_cols if x not in ['orderlabel']])


# In[28]:

sample = train_df[use_cols]


# In[29]:

not_rename = [
        'orderid', 'uid', 'hotelid', 'basicroomid', 'hotel_roomid', 'roomid',
        'orderlabel'
    ]


# In[30]:

sample.rename_axis({x:'order_'+ x for x in use_cols if x not in not_rename}, inplace=True, axis='columns')


# In[31]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

