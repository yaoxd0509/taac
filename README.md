# Tianchi News Recommendation

这个项目是天池新闻推荐任务的 ItemCF baseline。原始代码来自 `code.md`，现在已经拆分为多个 Python 文件，便于维护和运行。

## 目录结构

```text
.
|-- code.md                  # 原始 Markdown 代码
|-- config.py                # 路径配置
|-- data_loader.py           # 数据读取函数
|-- recall.py                # ItemCF 相似度计算和召回逻辑
|-- submit.py                # 提交文件生成函数
|-- utils.py                 # 通用工具函数
|-- main.py                  # 主运行入口
|-- dataset/                 # 数据集目录
`-- temp_results/            # 中间结果和提交文件输出目录
```

## 文件说明

### `config.py`

项目路径配置文件：

- `data_path = './dataset/'`：数据集所在目录
- `save_path = './temp_results/'`：相似度文件和提交文件输出目录

### `data_loader.py`

负责读取点击日志数据：

- `get_all_click_sample(data_path, sample_nums=10000)`：从训练集中抽样部分用户数据，便于调试
- `get_all_click_df(data_path='./dataset/', offline=True)`：读取点击日志；`offline=False` 时会合并训练集和测试集点击日志

### `recall.py`

负责 ItemCF 召回：

- `get_user_item_time(click_df)`：生成用户点击文章序列
- `get_item_topk_click(click_df, k)`：获取点击量最高的文章
- `itemcf_sim(df)`：计算文章之间的 ItemCF 相似度，并保存为 `temp_results/itemcf_i2i_sim.pkl`
- `item_based_recommend(...)`：基于文章相似度为用户生成召回结果

### `submit.py`

负责生成提交文件：

- `submit(recall_df, topk=5, model_name=None)`：将召回结果整理成提交格式，并保存到 `temp_results/`

### `utils.py`

通用工具函数：

- `reduce_mem(df)`：降低 DataFrame 内存占用

### `main.py`

项目主入口，串联完整流程：

1. 分别读取训练集和测试集点击日志
2. 使用训练集计算 ItemCF 文章相似度
3. 保存 `itemcf_i2i_sim.pkl`
4. 只为测试集用户生成召回文章
5. 整理测试集用户召回结果
6. 生成提交文件

## 数据准备

请将数据文件放在 `dataset/` 目录下。当前代码会用到：

```text
dataset/train_click_log.csv
dataset/testA_click_log.csv
```

当前目录中还包含：

```text
dataset/articles.csv
dataset/articles_emb.csv
```

这两个文件当前 baseline 代码暂未使用。

## 环境依赖

需要 Python 3，并安装以下依赖：

```bash
pip install pandas numpy tqdm
```

## 运行方式

在项目根目录执行：

```bash
python main.py
```

运行完成后会在 `temp_results/` 目录下生成：

```text
itemcf_i2i_sim.pkl
itemcf_baseline_MM-DD.csv
```

其中 `MM-DD` 是运行当天的日期。

## 注意事项

- `main.py` 会直接执行完整召回流程，数据量较大时运行时间会比较长。
- `temp_results/itemcf_i2i_sim.pkl` 是运行时生成的中间文件，不需要提前准备。
- 如果只想测试读取数据或单个函数，可以在其他脚本中单独导入对应模块。
