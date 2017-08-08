
# coding: utf-8

# In[1]:

import os
import sys
from datetime import datetime
from os import remove
from os.path import join, abspath, exists
from warnings import warn
from subprocess import check_output


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

feature_path = join(file_dir, 'all_feature.pkl')
print(datetime.now(), 'begin combine', feature_path)


# In[4]:

order_path = join(file_dir, 'order_feature.pkl')

hotel_path = join(file_dir, 'hotel_feature.pkl')

basic_path = join(file_dir, 'basic_room_feature.pkl')

room_path = join(file_dir, 'room_feature.pkl')

hotel_room_path = join(file_dir, 'hotel_room_feature.pkl')

user_path = join(file_dir, 'user_feature.pkl')


# In[5]:

# order_path = join(file_dir, 'select_all_feature.pkl')


# In[6]:

if exists(abspath(feature_path)):
    remove(abspath(feature_path))


# In[7]:

print(check_output(['ln', '-s', abspath(order_path), abspath(feature_path)]))


# In[ ]:

# order_df = pd.read_pickle(order_path)


# In[ ]:

t = 'hotelid'


# In[ ]:

p = hotel_path


# In[ ]:

def join_df(t, p, order_df):
    df = pd.read_pickle(p).set_index(t)
    order_df = order_df.join(df, on=t)
    return order_df


# In[ ]:

# order_df = join_df('hotelid', hotel_path, order_df)

# order_df = join_df('basicroomid', basic_path, order_df)

# order_df = join_df('roomid', room_path, order_df)

# order_df = join_df('hotel_roomid', hotel_room_path, order_df)

# order_df = join_df('uid', user_path, order_df)


# In[ ]:

# print(order_df.shape)


# In[27]:

# order_df.to_pickle(feature_path)

print(datetime.now(), 'save to', feature_path)

