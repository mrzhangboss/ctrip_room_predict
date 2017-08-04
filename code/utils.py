import pandas as pd
import numpy as np
import scipy as sp
from warnings import warn
from os.path import join
from datetime import datetime


def _press_df(predict, c, fillna, not_null, typ):
    if fillna:
        predict[c] = predict[c].astype(typ)
    else:
        predict.loc[not_null, c] = predict.loc[not_null, c].astype(typ)
    return predict

def press_date(predict, columns=None, fillna=False, excluded=None):
    columns = predict.columns if columns is None else columns
    excluded = [] if excluded is None else excluded
    for c in (x for x in columns if x not in excluded):
        if fillna and predict[c].dtype not in (np.float32, np.float32, np.float64, np.float128):
            predict[c] = predict[c].fillna(0)
        not_null = predict[c].notnull()
        c_max = max(predict[c].max(), abs(predict[c].min()))
        if np.isnan(c_max):
            warn('column {} is null; please check'.format(c))
            continue
        if c_max > 2147483647:
            warn('column {} bigger than int32.max:{}; please check'.format(c, c_max))
        elif c_max < 1:
            warn('column  {} is  may be error when meet percent max:{}'.format(c, c_max))
        if c_max < 256:
            if predict[c].dtype in (np.float32, np.float32, np.float64, np.float128):
                predict = _press_df(predict, c, True, not_null, np.float16)
            else: predict = _press_df(predict, c, fillna, not_null, np.int8)
        elif c_max < 32767:
            if predict[c].dtype in (np.float32, np.float32, np.float64, np.float128):
                predict = _press_df(predict, c, True, not_null, np.float16)
            else: predict = _press_df(predict, c, fillna, not_null, np.int16)
        else:
            if predict[c].dtype in (np.float32, np.float32, np.float64, np.float128):
                predict = _press_df(predict, c, True, not_null, np.float32)
            else: predict = _press_df(predict, c, fillna, not_null, np.int32)
    return predict

def extract_int_from_str(df, column):
    if df[column].dtype == object:
        df[column] = df[column].astype(str)
    df[column] = df[column].str.extract('(\d+)')
    if df[column].isnull().sum() > 0:
        df[column] = df[column].astype(np.float64)
    else:
        df[column] = df[column].astype(np.int64)
    df = press_date(df, [column])
    return df

def split_user_base_feature(tdf, user_cols, timespan):
    """提取用户基本特征"""
    user_feature = tdf[user_cols].drop_duplicates()
    if (tdf[user_cols].drop_duplicates('uid').shape[0] - user_feature.shape[0]) != 0:
        raise SystemError('error user feture, need check train {} user data'.format(timespan))
    filename = join('..', 'dataset', timespan[-2:], 'user_base_feture.pkl')
    user_feature.to_pickle(filename)
    user_feature.info()
    
def print_shape(train_df, names):
    result = []
    print(datetime.now())
    print('--'* 10)
    for t in names:
        basic_shape = train_df[t].unique().shape[0]
        result.append(basic_shape)
        print('{} uniuqe shape'.format(t), basic_shape)
    print('--'* 10)
    return result

def extract_feature_count(t, f, train_df, sample):
    """ 分类计数统计 """
    use_cols = [t, f]
    name = '{}__{}_count'.format(t, f)
    tdf = train_df[use_cols].drop_duplicates()
    ntdf = tdf.groupby(t)[f].count()
    ntdf.name = name
    sample = sample.join(ntdf, on=t)
    sample = press_date(sample, [name])
    return sample

def extract_value_describe_feature(t, f, train_df, sample, use_cols=None):
    """ 数值型统计特征 支持（'max', 'min', 'median', 'mean', 'std', 'nunique', 'count'）"""
    if train_df[f].dtype in (np.float16, np.float32):
        train_df[f] = train_df[f].astype(np.float64)
    if train_df[f].dtype in (np.int8, np.int16):
        train_df[f] = train_df[f].astype(np.int64)
    use_cols = use_cols if use_cols else ['max', 'min', 'median', 'mean', 'std', 'nunique', 'count']
    name_fmt = '{}__{}_{}'.format(t, f, '{}')
    df = pd.DataFrame()
    for i, c in enumerate(use_cols):
        name = name_fmt.format(c)
        df[name] = getattr(train_df.groupby(t)[f], c)()
    sample = sample.join(df, on=t)
    sample = press_date(sample, df.columns)
    train_df = press_date(train_df, [f])
    return sample

def get_corr(train_df, sample, t):
    old_cols = [c for c in train_df.columns if c!='orderlabel']
    tdf = train_df.join(sample.set_index(t), on=t)
    new_cols = [c for c in tdf.columns if c not in old_cols]
    return tdf[new_cols].corr()

def extract_lastord_is_nan(hotel_lastord, sample, t, f):
    name = '{}_is_nan'.format(f)
    hotel_lastord[name] = np.nan
    hotel_lastord.loc[hotel_lastord[f].isnull(), name] = 1
    hotel_lastord_is_nan_count = hotel_lastord.groupby(t)[name].count()
    hotel_lastord_is_nan_count.loc[hotel_lastord_is_nan_count>1] = 1
    sample = sample.join(hotel_lastord_is_nan_count, on=t)
    sample = press_date(sample, [hotel_lastord_is_nan_count.name])
    return sample

def extract_is_lastord(hotel_lastord, sample, t, f):
    name = '{}_is_lastord'.format(t)
    hotel_lastord[name] = (hotel_lastord[f] == hotel_lastord[t]).astype(np.int8)
    hotel_lastord.loc[hotel_lastord[name]==0, name] = np.nan
    hotel_is_lastord_count = hotel_lastord.groupby(t)[name].count()
    hotel_is_lastord_count.loc[hotel_is_lastord_count>1] = 1
    sample = sample.join(hotel_is_lastord_count, on=t)
    sample = press_date(sample, [hotel_is_lastord_count.name])
    return sample

def add_column(train_df, sample, t, f, need_warning=True):
    df = train_df[[t, f]].drop_duplicates([t, f])
    if df.shape[0] != train_df.drop_duplicates(t).shape[0] and need_warning:
        warn('df.shape[0] != train_df.drop_duplicates(t).shape[0]')
    name = '{}_{}'.format(t, f)
    df = df.set_index(t)
    df.columns = [name]
    sample = sample.join(df, on=t)
    sample = press_date(sample, [name])
    return sample