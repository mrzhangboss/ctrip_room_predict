
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


# In[2]:

dir_arg = sys.argv[1]
if dir_arg == '-f':
    file_dir = join('..', 'dataset', '11')
else:
    file_dir = join('..', 'dataset',  dir_arg)


# In[25]:

feature_path = join(file_dir, 'all_feature.pkl')
print(datetime.now(), 'begin combine', feature_path)

order_path = join(file_dir, 'order_feature.pkl')

hotel_path = join(file_dir, 'hotel_feature.pkl')

basic_path = join(file_dir, 'basic_room_feature.pkl')

room_path = join(file_dir, 'room_feature.pkl')

hotel_room_path = join(file_dir, 'hotel_room_feature.pkl')

user_path = join(file_dir, 'user_feature.pkl')


# In[16]:

order_df = pd.read_pickle(order_path)


# In[7]:

t = 'hotelid'


# In[8]:

p = hotel_path


# In[14]:

def join_df(t, p, order_df):
    df = pd.read_pickle(p).set_index(t)
    order_df = order_df.join(df, on=t)
    return order_df


# In[18]:

order_df = join_df('hotelid', hotel_path, order_df)


# In[19]:

order_df = join_df('basicroomid', basic_path, order_df)


# In[20]:

order_df = join_df('roomid', room_path, order_df)


# In[21]:

order_df = join_df('hotel_roomid', hotel_room_path, order_df)


# In[22]:

order_df = join_df('uid', user_path, order_df)


# In[24]:

print(order_df.shape)


# In[27]:

order_df.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

