import math
import pickle
from collections import defaultdict
from tqdm import tqdm

from config import save_path


# 根据点击时间生成用户的文章点击序列：{user1: [(item1, time1), (item2, time2), ...]}。
def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp')
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict


# 获取点击次数最多的文章。
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def itemcf_sim(df):
    """
    计算文章之间的 ItemCF 相似度矩阵。

    df: 点击日志数据。
    return: 文章相似度字典，格式为 {item_i: {item_j: score}}。
    """
    
    # {
    # 1: [(101, 1000), (102, 1005), (103, 1010)],
    # 2: [(101, 2000), (102, 2005)]
    # }
    user_item_time_dict = get_user_item_time(df)
    
    # 统计物品共现次数并计算相似度。
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 这里保留原始 ItemCF 逻辑，后续可以继续加入时间权重等优化。
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if(i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)
                
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            # 归一化
            # 为什么要除以这个？避免热门文章和所有文章都相似
            # 因为热门文章点击量很大，容易和很多文章一起出现。如果不处理，热门文章会和所有文章都显得“很相似”
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将相似度矩阵保存到本地，供后续召回阶段读取。
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    # {
    # 101: {
    #     102: 0.8,
    #     103: 0.3
    # },
    # 102: {
    #     101: 0.8
    # }
    # }
    return i2i_sim_

# 基于文章的 ItemCF 召回。
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
    为单个用户生成基于 ItemCF 的召回结果。

    user_id: 用户 ID。
    user_item_time_dict: 用户历史点击序列，格式为 {user: [(item, time), ...]}。
    i2i_sim: 文章相似度矩阵。
    sim_item_topk: 每篇历史点击文章取相似度最高的前 k 篇文章。
    recall_item_num: 最终召回文章数量。
    item_topk_click: 热门文章列表，用于召回数量不足时补全。
    return: 召回文章列表，格式为 [(item, score), ...]。
    """

    # 用户看过文章 A、B、C
    # ↓
    # 找出和 A、B、C 相似的文章
    #     ↓
    # 去掉用户已经看过的文章
    #     ↓
    # 按相似度分数排序
    #     ↓
    # 如果数量不够，用热门文章补齐
    #     ↓
    # 返回前 N 篇文章
    
    # 获取用户历史点击文章，避免重复推荐。
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim.get(i, {}).items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
                
            item_rank.setdefault(j, 0)
            # 针对一个物品的推荐分数，而不是最终分数，需要累加，比如历史里面A和B都跟C相似，那C的分数可以累加
            item_rank[j] +=  wij
    
    # 召回数量不足时，用热门文章补全。
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank: # 补全文章不应出现在已有召回列表中。
                continue
            item_rank[item] = - i - 100 # 给热门补全文章一个较低分数。
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank
