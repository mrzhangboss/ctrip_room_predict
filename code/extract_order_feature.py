
# coding: utf-8

# In[1]:

import sys
import gc
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

idf = pd.read_csv('models/06-1736-importance.txt', index_col=[0])

nidf = idf.loc[idf.index.str.extract('^(orderid)').notnull()]

nidf

nidf.loc[nidf['0']==0]

idf.loc[[x for x in nidf.index if x.startswith('basicroomid__basic_30days_realratio')]]

nidf.head()


# In[5]:

feature_path = join(file_dir, 'order_feature.pkl')
print(datetime.now(), 'begin', feature_path)


# In[6]:

hotel_path = join(file_dir, 'hotel_feature.pkl')
basic_path = join(file_dir, 'basic_room_feature.pkl')
room_path = join(file_dir, 'room_feature.pkl')
hotel_room_path = join(file_dir, 'hotel_room_feature.pkl')
user_path = join(file_dir, 'user_feature.pkl')


# In[7]:

not_rename = []


# In[8]:

def join_df(t, p, order_df):
    df = pd.read_pickle(p).set_index(t)
    not_rename.extend(list(df.columns))
    order_df = order_df.join(df, on=t)
    return order_df


# In[9]:

train_df = join_df('hotelid', hotel_path, train_df)
train_df = join_df('basicroomid', basic_path, train_df)
train_df = join_df('roomid', room_path, train_df)
train_df = join_df('hotel_roomid', hotel_room_path, train_df)
train_df = join_df('uid', user_path, train_df)

gc.collect()


# In[23]:

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

# In[22]:

train_df=df_median(train_df)
train_df=df_min(train_df)
train_df=df_min_orderid(train_df)


# In[24]:

train_df['basicroomid_price_rank'] = train_df['price_deduct'].groupby([train_df['orderid'], train_df['basicroomid']]).rank()


# In[25]:

train_df = df_roomrank_mean(train_df)


# In[26]:

train_df["orderid_price_deduct_min_rank"] = train_df['orderid_price_deduct_min'].groupby(train_df['orderid']).rank()


# In[27]:

# train_df = press_date(train_df, ['order_basicroomid_price_rank'])


# ### 上次订购的价格和当时最低价的比

# In[28]:

for i in range(2, 9):
    t = 'roomservice_%d' % i
    if i != 7:
        train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
        train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[29]:

for i in range(2, 5):
    t = 'roomtag_%d' % i
    train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
    train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[30]:

for t in ['rank', 'star', 'basicroomid', 'hotelid']:
    train_df[t + '_is_equal'] = (train_df[t] == train_df[t+'_lastord']).astype(np.int8)
    train_df.loc[train_df.orderdate_lastord.isnull(), t + '_is_equal'] = np.nan


# In[31]:

train_df['order_weekday'] = train_df.orderdate.dt.weekday

train_df['order_weekday_lastord'] = train_df.orderdate_lastord.dt.weekday


# In[32]:

train_df["this_is_basicroomid_price_deduct_min"] = (train_df["basicroomid_price_deduct_min"] == train_df.price_deduct).astype(np.int8)


# In[33]:

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
train_df.loc[(train_df.price_tail1==4)|(train_df.price_tail1==7), "price_tail1"]= 1
train_df.loc[(train_df.price_tail1!=4)&(train_df.price_tail1!=7), "price_tail1"]= 0


train_df["price_dx"] = train_df["price_deduct"] - train_df["price_last_lastord"] 

train_df["return_dx"] = train_df["returnvalue"] - train_df["return_lastord"]

train_df["price_ori"] = train_df["price_deduct"] + train_df["returnvalue"]


# ## 8/7号添加

# In[36]:

train_df['roomtag_3_1'] = train_df.roomtag_3 * train_df.roomtag_1


# #### 用户特征价格比较 

# ### orderid特征统计

# In[37]:

group = train_df[['orderid','price_deduct','returnvalue','basic_week_ordernum_ratio','basic_recent3_ordernum_ratio','basic_comment_ratio',
               'basic_30days_ordnumratio','basic_30days_realratio']].groupby('orderid')

group_min = group.min().reset_index()
group_min.columns = group_min.columns.map(lambda x: 'min_'+x if x!='orderid' else x)

group_max = group.max().reset_index()
group_max.columns = group_max.columns.map(lambda x: 'max_'+x if x!='orderid' else x)


group2 = train_df[['orderid','basicroomid','room_30days_ordnumratio','room_30days_realratio']].groupby(['orderid','basicroomid'])

group2_min = group2.min().reset_index().rename(columns={'room_30days_ordnumratio':'min_room_30days_ordnumratio','room_30days_realratio':'min_room_30days_realratio'})
group2_max = group2.max().reset_index().rename(columns={'room_30days_ordnumratio':'max_room_30days_ordnumratio','room_30days_realratio':'max_room_30days_realratio'})

train_df = pd.merge(train_df,group_min,how='left',on='orderid')
train_df = pd.merge(train_df,group_max,how='left',on='orderid')
train_df = pd.merge(train_df,group2_min,how='left',on=['orderid','basicroomid'])
train_df = pd.merge(train_df,group2_max,how='left',on=['orderid','basicroomid'])


# In[38]:

train_df['user_price_deduct_user_maxprice_1week']=train_df.price_deduct-train_df.user_maxprice_1week
train_df['user_price_deduct_user_minprice_1week']=train_df.price_deduct-train_df.user_minprice_1week
train_df['user_price_deduct_user_maxprice_1month']=train_df.price_deduct-train_df.user_maxprice_1month
train_df['user_price_deduct_user_minprice_1month']=train_df.price_deduct-train_df.user_minprice_1month
train_df['user_price_deduct_user_maxprice_3month']=train_df.price_deduct-train_df.user_maxprice_3month
train_df['user_price_deduct_user_minprice_3month']=train_df.price_deduct-train_df.user_minprice_3month
train_df['price_deduct_diff_up2std']=train_df.price_deduct-train_df.user_avgprice+2*train_df.user_stdprice
train_df['price_deduct_diff_down2std']=train_df.price_deduct-train_df.user_avgprice-2*train_df.user_stdprice

train_df['price_diff_order_min'] = train_df['price_deduct']-train_df['min_price_deduct']
train_df['price_diff_user_med_1week'] = train_df['price_deduct']-train_df['user_medprice_1week']
train_df['price_diff_user_med_1month'] = train_df['price_deduct']-train_df['user_medprice_1month']
train_df['price_diff_user_med_3month'] = train_df['price_deduct']-train_df['user_medprice_3month']


# In[39]:

train_df.user_avgadvanceddate=train_df.user_avgadvanceddate.apply(round).astype('int')#convert user_avgadvanceddata to int,so we can get real data(orderdate+adv_data)
train_df['is_holiday']=(((train_df.order_weekday+train_df.user_avgadvanceddate)%7==0)|((train_df.order_weekday+train_df.user_avgadvanceddate)%7==6)).astype(np.int8)


# In[40]:

train_df['basicroomid_roomid_price_rank']=train_df.groupby(['orderid','basicroomid'])['returnvalue'].rank(method='max')
train_df['basicroomid_roomid_price_ismin']=(train_df['basicroomid_roomid_price_rank']==1).astype(np.int8)


# In[41]:

train_df['orderid_roomid_price_rank']=train_df.groupby(['orderid'])['returnvalue'].rank(method='max')
train_df['orderid_roomid_price_ismin']=(train_df['orderid_roomid_price_rank']==1).astype(np.int8)


# In[42]:

t=train_df[['orderid','basicroomid','roomid']].drop_duplicates()[['orderid','basicroomid']]     ##how many roomid in each (orderid,basicroomid)
t['basicroomid_roomid_cnt']=1
t=t.groupby(['orderid','basicroomid']).agg('sum').reset_index()
train_df=pd.merge(train_df,t,on=['orderid','basicroomid'],how='left')


# In[43]:

train_df['basicroomid_roomid_rank1']=train_df.groupby(['orderid','basicroomid'])['rank'].rank(method='max')
# train_df['basicroomid_roomid_rank1_ismin']=(train_df['basicroomid_roomid_price_rank']==1).astype(np.int8)
train_df['basicroomid_roomid_rank1_rate']=train_df.basicroomid_roomid_rank1.astype('float')/train_df.basicroomid_roomid_cnt


# In[44]:

train_df['basicroomid_hotel_basic_count_rate'] = train_df.basicroomid_roomid_rank1.astype('float')/train_df.hotelid__basicroomid_count


# In[45]:

train_df['true_value_basic_rank']=train_df.groupby(['orderid','basicroomid'])['price_ori'].rank(method='max')
train_df['price_deduct_basic_rank']=train_df.groupby(['orderid','basicroomid'])['price_deduct'].rank(method='max')


# In[46]:

train_df['true_value_rank']=train_df.groupby(['orderid'])['price_ori'].rank(method='max')
train_df['price_deduct_rank']=train_df.groupby(['orderid'])['price_deduct'].rank(method='max')
train_df['basic_minarea_rank']=train_df.groupby(['orderid'])['basic_minarea'].rank(method='max')
train_df['basic_maxarea_rank']=train_df.groupby(['orderid'])['basic_maxarea'].rank(method='max')


# In[47]:

train_df['this_last_roomservice_2_gap']=train_df.roomservice_2-train_df.roomservice_2_lastord
train_df['this_last_roomservice_3_gap']=train_df.roomservice_3-train_df.roomservice_3_lastord
train_df['this_last_roomservice_4_gap']=train_df.roomservice_4-train_df.roomservice_4_lastord
train_df['this_last_roomservice_5_gap']=train_df.roomservice_5-train_df.roomservice_5_lastord
train_df['this_last_roomservice_6_gap']=train_df.roomservice_6-train_df.roomservice_6_lastord
train_df['this_last_roomservice_8_gap']=train_df.roomservice_8-train_df.roomservice_8_lastord
train_df['this_last_roomtag_4_gap']=train_df.roomtag_4-train_df.roomtag_4_lastord
# train_df['this_last_roomtag_5_gap']=train_df.roomtag_5-train_df.roomtag_5_lastord


# In[48]:

train_df['user_maxprice_hotel_minprice_lastord_gap']=train_df.user_maxprice-train_df.hotel_minprice_lastord
train_df['user_maxprice_basic_minprice_lastord_gap']=train_df.user_maxprice-train_df.basic_minprice_lastord
train_df['user_minprice_hotel_minprice_lastord_gap']=train_df.user_minprice-train_df.hotel_minprice_lastord
train_df['user_minprice_basic_minprice_lastord_gap']=train_df.user_minprice-train_df.basic_minprice_lastord
train_df['user_stdprice_hotel_minprice_lastord_gap']=train_df.user_stdprice-train_df.hotel_minprice_lastord
train_df['user_stdprice_basic_minprice_lastord_gap']=train_df.user_stdprice-train_df.basic_minprice_lastord


# In[49]:

train_df['user_price_deduct_user_avgdealpriceholiday']=train_df.price_deduct-train_df.user_avgdealpriceholiday
train_df['user_price_deduct_user_avgdealpriceworkday']=train_df.price_deduct-train_df.user_avgdealpriceworkday
train_df['user_price_deduct_user_avgdealprice']=train_df.price_deduct-train_df.user_avgdealprice
train_df['user_price_deduct_user_avgprice_1week']=train_df.price_deduct-train_df.user_avgprice_1week
train_df['user_price_deduct_user_avgprice_1month']=train_df.price_deduct-train_df.user_avgprice_1month
train_df['user_price_deduct_user_avgprice_3month']=train_df.price_deduct-train_df.user_avgprice_3month


# In[50]:

train_df['this_price_last_avgprice_gap']=train_df.price_deduct-train_df.user_avgprice
train_df['this_price_last_maxprice_gap']=train_df.price_deduct-train_df.user_maxprice
train_df['this_price_last_minprice_gap']=train_df.price_deduct-train_df.user_minprice


# In[51]:

train_df["price_star"]=train_df["price_deduct"]/(train_df["star"])
train_df["price_minarea"]=train_df["price_deduct"]/(train_df["basic_minarea"]-1)

train_df["star_dif"]=train_df["user_avgstar"]-train_df["star"]

train_df["price_ave_dif_rt"]=train_df["price_deduct"]/train_df["user_avgdealprice"]
train_df["price_ave_star_dif"]=train_df["price_deduct"]/train_df["user_avgprice_star"]
train_df["price_h_w_rt"]=train_df["user_avgdealpriceholiday"]/train_df["user_avgdealpriceworkday"]

train_df["price_ave_dif"] = train_df["price_deduct"] - train_df["user_avgdealprice"]


# In[52]:

train_df["order_hotel_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_rt"]=train_df["price_last_lastord"]/train_df["basic_minprice_lastord"]
train_df["order_hotel_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["hotel_minprice_lastord"]
train_df["order_basic_last_price_min_dif"]=train_df["price_last_lastord"]-train_df["basic_minprice_lastord"]


# In[53]:

train_df = press_date(train_df, ['order_hotel_last_price_min_rt', 'order_basic_last_price_min_rt', 'order_hotel_last_price_min_dif', 'order_basic_last_price_min_dif'])


# #### 价格特征综合

# In[54]:

tt=train_df.groupby('orderid')['price_deduct'].agg('median').reset_index()
tt.rename(columns={'price_deduct':'price_deduct_median'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['returnvalue'].agg('median').reset_index()
tt.rename(columns={'returnvalue':'returnvalue_median'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['price_ori'].agg('median').reset_index()
tt.rename(columns={'price_ori':'price_ori_median'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['basic_minarea'].agg('median').reset_index()
tt.rename(columns={'basic_minarea':'basic_minarea_median'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['basic_maxarea'].agg('median').reset_index()
tt.rename(columns={'basic_maxarea':'basic_maxarea_median'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['price_deduct'].agg('std').reset_index()
tt.rename(columns={'price_deduct':'price_deduct_std'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['returnvalue'].agg('std').reset_index()
tt.rename(columns={'returnvalue':'returnvalue_std'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['price_ori'].agg('std').reset_index()
tt.rename(columns={'price_ori':'price_ori_std'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['basic_minarea'].agg('std').reset_index()
tt.rename(columns={'basic_minarea':'basic_minarea_std'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')

tt=train_df.groupby('orderid')['basic_maxarea'].agg('std').reset_index()
tt.rename(columns={'basic_maxarea':'basic_maxarea_std'},inplace=True)
train_df=pd.merge(train_df,tt,on='orderid',how='left')


# In[55]:

for c in ['basic_minarea_std', 'basic_maxarea_std']:
    train_df[c] = train_df[c].astype(np.float).replace([np.inf, -np.inf], np.nan)


# In[56]:

# train_df = press_date(train_df, ['basic_minarea_std', 'basic_maxarea_std'])


# In[57]:

train_df['price_deduct_std_rate']=train_df.price_deduct_std/train_df.basicroomid_price_deduct_median
train_df['returnvalue_std_rate']=train_df.returnvalue_std/train_df.returnvalue_median
train_df['price_ori_std_rate']=train_df.price_ori_std/train_df.price_ori_median
train_df['basic_minarea_std_rate']=train_df.basic_minarea_std/train_df.basic_minarea_median
train_df['basic_maxarea_std_rate']=train_df.basic_maxarea_std/train_df.basic_maxarea_median


# #### 价格交叉特征 

# ## 8.9 日添加

# In[77]:

train_df['basicroomid_roomid_rank1_rate_rank'] = train_df.basicroomid_roomid_rank1_rate.groupby(train_df['orderid']).rank()


# In[103]:

# train_df['true_value_rank_small_than_10'] = pd.cut(train_df['true_value_rank'], 10).astype('category').cat.codes


# In[84]:

# train_df['true_value_rank_small_than_10'] = (train_df['true_value_rank'] < 10).astype(np.int8)


# In[105]:

# train_df[['basicroomid_roomid_rank1_rate_rank', 'basicroomid_roomid_rank1_rate', 'true_value_rank_small_than_10', 'true_value_rank', 'orderlabel']].corr()


# In[112]:

# train_df['price_dif_cut'] = (train_df.price_dif < -2.6).astype(np.int8)


# In[115]:

# train_df['price_dif_cut'] = pd.cut(train_df.price_dif, 3).astype('category').cat.codes


# In[117]:

# train_df[['orderlabel', 'price_dif', 'price_dif_hotel', 'price_dif_cut']].corr()


# In[116]:

# train_df.basicroomid_roomid_rank1_rate.diff()


# In[118]:

# train_df.groupby(['orderlabel']).basicroomid_roomid_rank1_rate.count()


# ### 用户价格交叉特征 

# ### 用户特征 

# In[ ]:

train_df["user_roomservice_8_2ratio"]=1-train_df["user_roomservice_8_345ratio"]-train_df["user_roomservice_8_1ratio"]
train_df["user_roomservice_4_1ratio_3month"] = 1 - train_df["user_roomservice_4_0ratio_3month"] - train_df["user_roomservice_4_2ratio_3month"] - train_df["user_roomservice_4_3ratio_3month"] - train_df["user_roomservice_4_4ratio_3month"] - train_df["user_roomservice_4_5ratio_3month"]
train_df["user_roomservice_4_1ratio_1month"] = 1 - train_df["user_roomservice_4_0ratio_1month"] - train_df["user_roomservice_4_2ratio_1month"] - train_df["user_roomservice_4_3ratio_1month"] - train_df["user_roomservice_4_4ratio_1month"] - train_df["user_roomservice_4_5ratio_1month"]
train_df["user_roomservice_4_1ratio_1week"] = 1 - train_df["user_roomservice_4_0ratio_1week"] - train_df["user_roomservice_4_2ratio_1week"] - train_df["user_roomservice_4_3ratio_1week"] - train_df["user_roomservice_4_4ratio_1week"] - train_df["user_roomservice_4_5ratio_1week"]
train_df["user_roomservice_2_0ratio"]=1-train_df["user_roomservice_2_1ratio"]
train_df["user_roomservice_3_0ratio"]=1-train_df["user_roomservice_3_123ratio"]
train_df["user_roomservice_5_0ratio"]=1-train_df["user_roomservice_5_1ratio"]
train_df['user_roomservice_6_other_ratio']=1-train_df.user_roomservice_6_0ratio-train_df.user_roomservice_6_1ratio-train_df.user_roomservice_6_2ratio
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


# In[73]:

for i in [1,2,3,4,5,6,7,8,9,10,11]:
    train_df["order_ordertype_%s_num"%i] = train_df["ordertype_%s_ratio"%i] * train_df["user_ordernum"]
    del train_df["ordertype_%s_ratio"%i]

for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio"]:
    train_df[c]= train_df[c] * train_df["user_ordernum"]

for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week"]:
    train_df[c]= train_df[c] * train_df["user_ordnum_1week"]

for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month"]:
    train_df[c]= train_df[c] * train_df["user_ordnum_3month"]


# In[42]:

train_df['user_ordnum_post_3week'] = train_df.user_ordnum_1month - train_df.user_ordnum_1week


# In[43]:

train_df['user_ordnum_post_2month'] = train_df.user_ordnum_3month - train_df.user_ordnum_1month


# In[74]:

train_df['orderspan'] = (now_date - train_df['orderdate_lastord']).dt.days.astype(np.float16)


# In[75]:

train_df['orderhour'] = train_df['orderdate'].dt.hour.astype(np.int8)


# In[81]:

def get_main_types(train):
    train['idxmax_rs2'] = train[['user_roomservice_2_0ratio','user_roomservice_2_1ratio']].idxmax(axis=1)
    train['idxmax_rs3'] = train[['user_roomservice_3_0ratio','user_roomservice_3_123ratio']].idxmax(axis=1)
    train['idxmax_rs4'] = train[['user_roomservice_4_0ratio','user_roomservice_4_1ratio','user_roomservice_4_2ratio','user_roomservice_4_3ratio',
                                 'user_roomservice_4_4ratio','user_roomservice_4_5ratio']].idxmax(axis=1)
    train['idxmax_rs5'] = train[['user_roomservice_5_0ratio','user_roomservice_5_1ratio']].idxmax(axis=1)
    train['idxmax_rs6'] = train[['user_roomservice_6_0ratio','user_roomservice_6_1ratio','user_roomservice_6_2ratio']].idxmax(axis=1)
    train['idxmax_rs7'] = train[['user_roomservice_7_0ratio','user_roomservice_7_1ratio']].idxmax(axis=1)
    train['idxmax_rs8'] = train[['user_roomservice_8_1ratio','user_roomservice_8_2ratio','user_roomservice_8_345ratio']].idxmax(axis=1)

    train['maintype_rs2'] = train['idxmax_rs2'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs3'] = train['idxmax_rs3'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs4'] = train['idxmax_rs4'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs5'] = train['idxmax_rs5'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs6'] = train['idxmax_rs6'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs7'] = train['idxmax_rs7'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)
    train['maintype_rs8'] = train['idxmax_rs8'].apply(lambda x: int(x[19:20]) if pd.notnull(x) else x)

    train['ismaintype_rs2'] = (train['roomservice_2']-train['maintype_rs2']).map({0:1,-1:0,1:0})
    train['roomservice_3'] = train['roomservice_3'].map({0:0,1:1,2:1,3:1})
    train['ismaintype_rs3'] = (train['roomservice_3']-train['maintype_rs3']).map({0:1,-1:0,1:0})
    train['ismaintype_rs4'] = (train['roomservice_4']-train['maintype_rs4']).map({0:1,-1:0,1:0,-2:0,2:0,-3:0,3:0,-4:0,4:0,-5:0,5:0})
    train['ismaintype_rs5'] = (train['roomservice_5']-train['maintype_rs5']).map({0:1,-1:0,1:0})
    train['ismaintype_rs6'] = (train['roomservice_6']-train['maintype_rs6']).map({0:1,-1:0,1:0,-2:0,2:0})
    train['ismaintype_rs7'] = (train['roomservice_7']-train['maintype_rs7']).map({0:1,-1:0,1:0})
    train['roomservice_8'] = train['roomservice_8'].map({1:1,2:2,3:3,4:3,5:3})
    train['ismaintype_rs8'] = (train['roomservice_8']-train['maintype_rs8']).map({0:1,-1:0,1:0,-2:0,2:0})
    train.drop(['idxmax_rs%d' % x for x in range(2, 9)] + ['maintype_rs%d' % x for x in range(2, 9)], axis=1, inplace=True)


# In[80]:

get_main_types(train=train_df)


# ## 交叉特征

# In[76]:

train_df['rank_roomservice_8'] = (
    train_df['roomservice_8'].astype(str) +
    train_df['rank'].astype(str)).astype('category').cat.codes


# In[ ]:

not_use_in_sample_cols = ['orderdate', 'orderdate_lastord',
                          # 'user_roomservice_6_1ratio',
#                          'user_roomservice_5_1ratio', 'user_roomservice_2_1ratio',
#                          'user_roomservice_8_345ratio', 'user_roomservice_4_5ratio_1week',
#                          'user_avgroomarea', 'user_roomservice_4_0ratio',
#                          'user_roomservice_4_0ratio_3month', 'min_returnvalue', 'min_basic_week_ordernum_ratio',
#                          'min_basic_recent3_ordernum_ratio', 'user_roomservice_4_5ratio_3month',
#                          'user_roomservice_4_5ratio_3month', 'user_roomservice_4_4ratio_3month',
#                          'min_basic_comment_ratio', 'min_basic_30days_ordnumratio', 'basic_week_ordernum_ratio',
#                          'basic_recent3_ordernum_ratio', 'basic_comment_ratio',
                         # order last id 
                          'hotelid_lastord', 'roomid_lastord', 'basicroomid_lastord',
                         ]


# In[77]:

use_cols = [x for x in train_df.columns if x not in not_use_in_sample_cols]


# In[95]:

train_df = press_date(train_df, [x for x in use_cols if x not in ['orderlabel']])


# In[96]:

sample = train_df[use_cols]


# In[97]:

not_rename  += [
        'orderid', 'uid', 'hotelid', 'basicroomid', 'hotel_roomid', 'roomid',
        'orderlabel'
    ]


# In[98]:

sample.rename_axis({x:'orderid_'+ x for x in use_cols if x not in not_rename}, inplace=True, axis='columns')


# In[99]:

sample.shape


# In[100]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

