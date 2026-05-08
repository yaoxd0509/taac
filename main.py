import collections
import time

import pandas as pd
from tqdm import tqdm

from config import data_path
from data_loader import get_all_click_df
from recall import get_item_topk_click, get_user_item_time, item_based_recommend, itemcf_sim
from submit import submit
from utils import timer


total_start_time = time.time()

with timer('读取点击数据'):
    # 训练集用于计算文章相似度，测试集用于生成最终提交。
    trn_click_df = get_all_click_df(data_path, offline=True)
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()

with timer('ItemCF 相似度计算'):
    # 计算并保存 ItemCF 文章相似度矩阵。
    i2i_sim = itemcf_sim(trn_click_df)

with timer('用户召回'):
    user_recall_items_dict = collections.defaultdict(dict)

    # 只构建测试集用户历史点击序列。
    user_item_time_dict = get_user_item_time(tst_click)

    # 每篇历史点击文章取相似度最高的前 10 篇文章。
    sim_item_topk = 10

    # 每个用户最终召回 10 篇候选文章。
    recall_item_num = 10

    # 点击量最高的文章，用于召回不足时补全。
    item_topk_click = get_item_topk_click(trn_click_df, k=50)

    # 只为测试集用户生成召回结果。
    for user in tqdm(tst_users):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                            sim_item_topk, recall_item_num, item_topk_click)

with timer('召回结果整理'):
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    tst_recall = recall_df

with timer('生成提交文件'):
    submit(tst_recall, topk=5, model_name='itemcf_baseline')

total_elapsed = time.time() - total_start_time
print(f'[总耗时] {total_elapsed:.2f}s ({total_elapsed / 60:.2f}min)')
