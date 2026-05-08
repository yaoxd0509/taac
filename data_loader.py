import numpy as np
import pandas as pd


# Debug 模式：从训练集中抽取一部分用户数据用于调试。
def get_all_click_sample(data_path, sample_nums=10000):
    """
    从训练集中采样一部分用户数据用于调试。

    data_path: 原始数据存储路径。
    sample_nums: 采样用户数量。
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 读取点击数据。
# offline=True 时只读取训练集；offline=False 时合并训练集和测试集点击日志，用于线上提交。
def get_all_click_df(data_path='./dataset/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = pd.concat([trn_click, tst_click])
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click
