
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


# In[5]:

train_df['orderspan'] = (now_date - train_df['orderdate_lastord']).dt.days.astype(np.float16)


# In[6]:

train_df['orderhour'] = train_df['orderdate'].dt.hour.astype(np.int8)


# In[12]:

order_cols = ['orderhour', 'orderid', 'uid', 'hotelid', 'basicroomid', 'hotel_roomid', 'roomid', 'orderlabel',
             'orderspan']


# In[13]:

room_service = ['roomservice_%d' % i for i in range(1, 9)]


# In[14]:

room_tag = ['roomtag_%d' % i for i in range(1, 5)]


# In[15]:

total_cols = list(chain(order_cols, room_service, room_tag))


# In[16]:

sample = train_df[total_cols]


# In[17]:

sample.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

