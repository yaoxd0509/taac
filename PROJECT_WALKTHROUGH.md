# TX-Game 项目入门拆解笔记

这份文档是给刚接触推荐系统、也刚接触这个项目的人准备的。

目标不是“把所有代码逐行翻译一遍”，而是帮你先建立一个清晰的脑图：

1. 这个项目在做什么
2. 它的输入数据长什么样
3. 数据是怎么变成模型能吃的张量的
4. 模型每个模块分别在干什么
5. 训练和验证是怎么跑起来的
6. 作为初学者，你应该按什么顺序学
7. 这个项目未来有哪些值得改进的方向

如果你现在是小白，不要急着啃 `model.py`。  
这个项目最难的地方不是某一个函数，而是“数据表示 + 模型结构 + 推荐系统背景”三件事叠在一起。  
所以最好的方式是：先把整体流程看懂，再逐步下钻。

---

## 1. 这个项目到底在做什么

一句话概括：

- 这是一个 **点击后转化率预测** 项目。

更准确一点说：

- 用户先点了一个东西
- 系统要判断，这次点击之后，用户最终会不会转化
- 这个“转化”通常可以理解成购买、下单、注册、充值等目标行为

在推荐系统/广告系统里，常见有几种预估任务：

- CTR：点击率预测，预测“会不会点”
- CVR：转化率预测，预测“点了以后会不会转化”
- PCVR：post-click CVR，点击后的转化率预测
- CTCVR：从曝光开始的整体转化率预测

这个项目更接近：

- 已经发生了点击
- 然后预测点击后的转化

也就是 PCVR 风格任务。

### 1.1 一个非常生活化的例子

假设你打开一个电商 App：

- 你点进了一双鞋
- 平台已经知道你最近看过什么、点过什么、加购过什么
- 也知道你的用户画像，比如城市、年龄段、消费层级
- 还知道当前这双鞋的类目、品牌、价格段等信息

现在平台想回答一个问题：

- 你这次点进去之后，会不会买？

这个项目就是在做这件事。

所以你可以把整个项目理解成：

- 输入：一个用户 + 一个商品 + 用户历史行为
- 输出：一个分数，表示“转化可能性”

---

## 2. 项目的整体流程

先把这条主线背下来：

```text
Parquet 训练数据 + schema.json
        ->
dataset.py 读取并整理
        ->
batch 字典
        ->
trainer.py 组装成 ModelInput
        ->
model.py 中的 PCVRHyFormer 做前向计算
        ->
输出 logits
        ->
计算 loss / AUC / logloss
```

你后面看任何细节，只要开始混乱，就回到这条链路上来。

### 2.1 这几个文件分别负责什么

- `train.py`
  - 训练入口
  - 解析参数
  - 构造数据集、模型、训练器

- `dataset.py`
  - 读取 Parquet
  - 读取 `schema.json`
  - 把原始列整理成张量

- `model.py`
  - 定义模型结构
  - 包括特征 embedding、序列编码、query 生成、最终分类

- `trainer.py`
  - 训练循环
  - 验证逻辑
  - early stopping
  - checkpoint 保存

- `utils.py`
  - 日志
  - 随机种子
  - early stopping
  - focal loss

---

## 3. 先补一点推荐系统背景，不然后面很难看懂

如果你以前更熟悉 NLP 或传统机器学习，推荐系统里有几个概念需要先建立。

### 3.1 什么是“特征”

在推荐系统里，输入通常不是一句文本，而是很多字段。

比如用户侧：

- 性别
- 年龄段
- 城市等级
- 会员等级
- 消费水平

比如商品侧：

- 商品 id
- 类目 id
- 品牌 id
- 价格区间

这些字段统称为“特征”。

### 3.2 什么是离散特征和连续特征

离散特征
取值来自一个有限或可枚举的类别集合，值本质上是“标签/类别身份”，不是连续数值量。数值之间通常没有自然的加减乘除意义。

连续特征
取值来自一个连续数值空间，值本身表示大小、强弱、程度、距离等，数值运算和相对差异有实际意义。

一个很实用的判断方法是问自己两个问题：

这个数字是“编号”还是“测量值”？
对它做加减比较大小，有没有现实语义？
如果是“编号”，通常是离散特征。
如果是“测量值”，通常是连续特征。

离散特征：

- 性别
- 类目 id
- 品牌 id
- 商品 id

它们通常用整数 id 表示，需要做 embedding。

连续特征：

- 价格
- 统计值
- 画像分数
- 某种实数向量

它们通常是 float，直接输入线性层。

### 3.3 什么是行为序列

行为序列就是用户过去做过的事情。

例如：

- 最近浏览过的商品序列
- 最近点击过的广告序列
- 最近加购过的商品序列
- 最近购买过的商品序列

为什么序列重要？

因为推荐系统很看重“兴趣是动态变化的”。

比如：

- 你 3 个月前看过手机
- 但你最近 3 天都在看跑鞋

那当前更可能转化的，往往是跑鞋相关商品。

所以只看静态画像是不够的，历史行为也要建模。

### 3.4 什么是 sparse feature

推荐系统常说 sparse feature，就是高基数离散特征。

例如：

- 用户 id
- 商品 id
- 店铺 id
- 品牌 id

因为可能值特别多，所以一般用 embedding 表来查向量。

---

## 4. 这个项目的输入数据长什么样

### 4.1 大类划分

这个项目把输入拆成了几块：

- `user_int_feats`
  - 用户侧离散特征

- `item_int_feats`
  - 商品侧离散特征

- `user_dense_feats`
  - 用户侧连续特征

- `item_dense_feats`
  - 商品侧连续特征
  - 当前实现里基本为空

- `seq_a`
- `seq_b`
- `seq_c`
- `seq_d`
  - 四路行为序列

此外每路序列还带：

- `seq_x_len`
  - 有效长度

- `seq_x_time_bucket`
  - 时间差分桶结果**（比如10秒前看过和30天前看过）**

标签是：

- `label = (label_type == 2)`

也就是说，这个模型的输入不是一句话，而是一整套结构化字段。

### 4.2 一个样本的直观例子

假设一个训练样本表示：

- 用户：男性，25-34 岁，一线城市，会员等级 3
- 当前商品：运动鞋，品牌 A，价格中高
- `seq_a`：最近浏览过的商品
- `seq_b`：最近点击过的商品
- `seq_c`：最近加购过的商品
- `seq_d`：最近别的交互序列

目标：

- 预测这次点击之后是否会购买

---

## 5. `schema.json` 是什么，为什么这么重要

当前目录里没有 `schema.json` 文件，但代码强依赖它。

### 5.1 它从哪里来

在 `train.py` 里：

- 如果传了 `--schema_path`，就用你传入的路径
- 否则默认去 `data_dir/schema.json` 找

所以现实里一般有两种情况：

- 数据团队把 `schema.json` 和 Parquet 一起产出
- 训练时手工指定 `--schema_path`

### 5.2 为什么必须有它

因为 Parquet 里只有原始列，模型不知道：

- 哪些列属于用户离散特征
- 哪些列属于商品离散特征
- 哪些列属于用户连续特征
- 哪些列属于序列域
- 每个离散特征词表大小是多少
- 每个序列域中哪个特征是时间戳

而 `schema.json` 正是这份“说明书”。

你可以把它理解成：

- 数据结构合同
- 特征字典
- 训练数据的元信息配置

### 5.3 从代码反推，它大概长什么样

根据 `dataset.py` 的解析逻辑，它至少包含：

- `user_int`
- `item_int`
- `user_dense`
- `seq`

大致结构像这样：

```json
{
  "user_int": [[fid, vocab_size, dim], ...],
  "item_int": [[fid, vocab_size, dim], ...],
  "user_dense": [[fid, dim], ...],
  "seq": {
    "seq_a": {
      "prefix": "seq_a",
      "ts_fid": 123,
      "features": [[fid1, vocab1], [fid2, vocab2], ...]
    }
  }
}
```

其中：

- `fid`
  - feature id，特征编号

- `vocab_size`
  - 这个离散特征一共有多少种 id

- `dim`
  - 如果这个特征本身是多值字段，它展开后占多少列

- `prefix`
  - 告诉代码去 Parquet 里找哪些列

- `ts_fid`
  - 哪个特征是时间戳，用来做时间差分桶

### 5.4 为什么初学者一定要先弄懂 schema

因为你如果不懂：

- 原始数据如何映射到 `user_int_feats`
- 一条行为序列怎么从多列字段变成一个三维张量

那你后面看模型就一定是一头雾水。

可以这么说：

- `schema.json` + `dataset.py` 是这个项目最重要的入门点

---

## 6. dataset 层到底做了什么

这一层负责把“原始数据”变成“模型输入张量”。

### 6.1 `FeatureSchema` 是什么

在 `dataset.py` 里有个 `FeatureSchema` 类。

它本质上是在做一件事：

- 记录每个特征在“拼接后总向量”里的偏移位置

比如你有 3 个离散特征：

- 特征 A 占 1 维
- 特征 B 占 1 维
- 特征 C 占 3 维

那么拼起来之后总长度就是 5。  
这时 schema 会记住：

- A 在哪一段
- B 在哪一段
- C 在哪一段

这样模型之后才能知道：

- 某个 feature id 对应输入向量中的哪一段切片

### 6.2 训练集和验证集怎么切

`get_pcvr_data()` 会：

- 扫描所有 parquet 文件
- 列出每个文件的 row group
- 用前面大部分 row group 当训练
- 用尾部一部分 row group 当验证

也就是说，它不是完全随机地逐样本切分，而是按 parquet row group 分块切。

这种做法比较常见，因为：

- 实现简单
- 读取高效

但缺点也有：

- 如果数据有时间顺序或分布漂移，切分方式会影响验证结果

这个我们后面在“改进方向”里会提到。

### 6.3 一个 batch 最终长什么样

`PCVRParquetDataset._convert_batch()` 最后会返回一个字典，大概包含：

```python
{
  "user_int_feats": Tensor[B, U_dim],
  "user_dense_feats": Tensor[B, U_dense_dim],
  "item_int_feats": Tensor[B, I_dim],
  "item_dense_feats": Tensor[B, 0],
  "label": Tensor[B],
  "timestamp": Tensor[B],
  "user_id": list,
  "_seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"],
  "seq_a": Tensor[B, S_a, L_a],
  "seq_a_len": Tensor[B],
  "seq_a_time_bucket": Tensor[B, L_a],
  ...
}
```

### 6.4 最容易让新手困惑的一点：为什么序列是 `[B, S, L]`

这里必须专门解释。

很多人第一次看推荐系统序列会以为：

- 一条序列就是一个商品 id 序列
- 所以 shape 应该是 `[B, L]`

但这个项目不是这么设计的。

这里一条序列域在每个时间步上，可能不止一个字段，而是多个 side-info 字段。

例如某条浏览序列中的每个时间步可能同时有：

- 商品 id
- 类目 id
- 品牌 id

那么如果序列长度是 5，就不是单行：

```text
[101, 202, 333, 444, 0]
```

而是三行：

```text
商品 id: [101, 202, 333, 444, 0]
类目 id: [  7,   7,   9,   9, 0]
品牌 id: [ 18,  18,  31,  31, 0]
```

所以 shape 就是：

- `[B, S, L]`

其中：

- `B`：batch size
- `S`：这一条序列域在每个时间步上有多少个 side-info 特征
- `L`：序列长度

### 6.5 时间桶是怎么做的

项目里对每条序列还会做时间差建模。

过程是：

1. 取当前样本的 `timestamp`
2. 取序列中每个行为的时间戳
3. 计算时间差
4. 按 `BUCKET_BOUNDARIES` 做分桶
5. 得到 `seq_x_time_bucket`

为什么要这么做？

因为：

- 10 秒前看过一个商品
- 和 10 天前看过一个商品

对当前转化的影响往往很不一样。

一个简单例子：

- 当前时刻：100000
- 历史行为时刻：99990、99900、99000
- 时间差：10、100、1000
- 然后映射到几个 bucket id

模型后面再给这些 bucket id 查 embedding。

---

## 7. trainer 里又做了一次什么整理

`trainer.py` 会把 dataset 返回的 batch 字典再包装成一个 `ModelInput`。

这样模型拿到的输入结构更整齐：

```python
ModelInput(
    user_int_feats=...,
    item_int_feats=...,
    user_dense_feats=...,
    item_dense_feats=...,
    seq_data={
        "seq_a": ...,
        "seq_b": ...,
        "seq_c": ...,
        "seq_d": ...
    },
    seq_lens={
        "seq_a": ...,
        "seq_b": ...,
        "seq_c": ...,
        "seq_d": ...
    },
    seq_time_buckets={
        "seq_a": ...,
        "seq_b": ...,
        "seq_c": ...,
        "seq_d": ...
    }
)
```

这一层本身没有复杂算法，主要是为了让模型接口更清晰。

---

## 8. 你刚才问的核心问题：序列“单独编码”到底是什么意思

这个问题非常关键。

答案是：

- 是先把序列每个时间步变成 token
- 然后再把这一串 token 送入序列编码器

注意，不是：

- 整条序列直接压成一个 token

### 8.1 第一步：每个时间步先变成一个 token

模型里这个过程在：

- `PCVRHyFormer._embed_seq_domain()`

逻辑是：

1. 对每个 side-info 特征分别做 embedding
2. 在同一个时间步上，把这些 embedding 拼起来
3. 通过线性层投影到 `d_model`
4. 再可选地加上 time bucket embedding

原始输入：

- `[B, S, L]`

会变成：

- `[B, L, D]`

这里：

- `L` 是时间步数
- `D` 是 `d_model`

也就是说：

- 每个时间步最终会有一个向量表示
- 这个向量就是这个时间步的 token

### 8.2 举一个最简单的例子

假设某条浏览序列里，每个时间步有 3 个 side-info：

- 商品 id
- 类目 id
- 品牌 id

某一个时间步是：

- 商品 = 101
- 类目 = 7
- 品牌 = 18

那么模型会做：

- `E_item(101)` -> 一个向量
- `E_cate(7)` -> 一个向量
- `E_brand(18)` -> 一个向量

把这 3 个向量拼接后，再映射到一个统一维度，比如 64 维。

于是：

- 这一个时间步就变成了 1 个 64 维 token

如果整个序列长度是 5，就会得到：

- 5 个 token

所以更准确地说，不是：

- 一条序列变成一个 token

而是：

- 一条序列变成一串 token

### 8.3 第二步：再对这串 token 做“序列编码”

这一步才是你说的“单独编码”。

项目里支持 3 种序列编码器：

- `SwiGLUEncoder`
- `TransformerEncoder`
- `LongerEncoder`

它们的输入都是：

- `[B, L, D]`

输出也仍然是：

- `[B, L, D]`

区别在于：

- `SwiGLUEncoder`
  - 更轻量
  - 不做自注意力

- `TransformerEncoder`
  - 标准自注意力
  - 更强，但开销更大

- `LongerEncoder`
  - 面向更长序列
  - 会做 top-k 压缩

所以序列处理的完整链路应该这样理解：

```text
原始序列 id
-> 每个时间步做 token 化
-> 得到一串 token
-> 用序列编码器建模 token 之间的关系
-> 得到上下文化后的序列 token
```

这才是“单独编码”的真正含义。

---

## 9. 什么是 Hybrid Transformer

“Hybrid Transformer” 这个词你不用把它想得太玄。

它在这里的意思很朴素：

- 不是把所有输入都粗暴塞进同一种 Transformer
- 而是把多种适合业务的数据结构和模块组合起来

这个项目混合了这些东西：

- sparse embedding
- dense feature projection
- 多路序列编码
- query token 生成
- cross attention
- token mixing
- 最终分类头

所以它不是“标准 NLP Transformer 模板”，而是：

- Transformer 思想
- 加上推荐系统业务需要的定制结构

你可以把它理解为：

- 一个“混合式的、多输入源的序列预估模型”

### 9.1 一个容易理解的类比

把这个模型想象成“侦探破案”。

静态特征：

- 像案件背景资料

不同序列：

- 像不同来源的证词

query token：

- 像侦探提出的问题

cross attention：

- 像拿着问题去各份证词里找最相关的内容

RankMixer：

- 像把不同证词和背景资料综合讨论

最终分类器：

- 像侦探给出最终判断

这个类比虽然不严格，但很适合新手建立直觉。

---

## 10. 整个模型可以拆成 5 个模块

最推荐你的学习方式，就是把模型拆成 5 块看。

### 10.1 模块 1：NS Tokenizer

你可以先把 NS 理解成：

- non-sequence token
- 也就是非序列特征形成的 token

输入包括：

- 用户离散特征
- 商品离散特征
- 用户连续特征

输出是：

- 一组静态 token

### 10.2 为什么要把静态特征也变成 token

因为后面模型想统一处理各种信息。

如果静态特征还是一大坨平铺向量，而序列信息是 token，
那后面做交互就不方便。

所以项目先把静态特征也“token 化”。

### 10.3 当前有两种 NS tokenizer

- `GroupNSTokenizer`
- `RankMixerNSTokenizer`

#### GroupNSTokenizer 怎么理解

思路很简单：

- 先把一些相关的离散特征分组
- 每组里的特征先各自 embedding
- 然后拼起来
- 投影成 1 个 token

举例：

假设用户特征分成两组：

- U1：性别、年龄段
- U2：城市等级、消费水平、会员等级

那么：

- U1 组 -> 1 个 token
- U2 组 -> 1 个 token

于是用户侧就形成多个 token。

#### 这里的“1 个 token”到底是什么意思

很多初学者会把“1 个 token”误解成：

- 长度只有 1
- 里面只有 1 个数字

这两个理解都不对。

这里的“1 个 token”真正的意思是：

- 它在 **token 这个维度上占 1 个位置**
- 但这个位置里面装的是一个 **D 维向量**

如果 `d_model = 64`，那么：

- 1 个 token 的 shape 不是 `[1]`
- 而是 `[64]`

如果 batch size = 2，那么一个组投影出来的结果是：

- `[2, 1, 64]`

含义是：

- 2 个样本
- 每个样本这个组对应 1 个 token 位置
- 每个 token 是 64 维向量

再举一个更具体的数值例子。

假设：

- `emb_dim = 4`
- `d_model = 8`
- U1 组里有两个特征：
  - 性别
  - 年龄段

某个样本：

- 性别 = 男，对应 id = 2
- 年龄段 = 25-34，对应 id = 5

embedding 查表后，假设得到：

- 性别 embedding = `[0.2, 0.5, -0.1, 0.7]`
- 年龄 embedding = `[0.3, -0.4, 0.8, 0.1]`

先拼接：

```text
[0.2, 0.5, -0.1, 0.7, 0.3, -0.4, 0.8, 0.1]
```

这时 shape 是：

- `[8]`

如果这里的线性层刚好把 8 维投到 8 维，那它经过线性层和 LayerNorm 后，
还是会得到一个新的 8 维向量，例如：

```text
[0.6, -0.2, 1.1, 0.3, -0.7, 0.4, 0.0, 0.9]
```

这个 **8 维向量**，就是：

- U1 组对应的 **1 个 token**

所以你可以把“U1 组 -> 1 个 token”理解成：

- “把 U1 组压缩成一个向量摘要”

而不是：

- “U1 组只剩下一个数字”

再看多个组时会更清楚。

假设：

- U1 -> 1 个 token
- U2 -> 1 个 token
- 商品侧还有 I1、I2 两组 -> 各 1 个 token
- 用户 dense 特征再投成 1 个 token

那么最终静态 token 可能是：

- `ns_tokens: [B, 5, 64]`

这里第二维的 `5` 表示：

- 一共有 5 个 token 位置

而不是：

- 每个 token 长度为 5

所以：

- `1 个 token` 说的是 **token 个数**
- `64 维` 说的是 **每个 token 的向量维度**

#### RankMixerNSTokenizer 怎么理解

这个思路稍微不一样：

- 不强制“一组对应一个 token”
- 而是把所有组 embedding 先拼成一个大向量
- 再切成固定数量的块
- 每块再映射成一个 token

这样可以更灵活控制 token 数量。

如果你现在是初学者，可以先不用深究两者优劣，只要记住：

- 它们都在做一件事：把静态特征变成 token

### 10.4 模块 2：Sequence Token Embedding

这一块负责把每条序列从原始 id 变成一串 token。

输入：

- `seq_a`, `seq_b`, `seq_c`, `seq_d`

每一条序列先是：

- `[B, S, L]`

经过 `_embed_seq_domain()` 后变成：

- `[B, L, D]`

你一定要记住这个 shape 变化。

这是理解整个模型的第一道坎。

### 10.5 模块 3：MultiSeqQueryGenerator

这是这个项目比较有特点的地方。

很多 Transformer 或推荐模型会用固定的全局 token，比如：

- `[CLS]`

但这个项目不是这样。

它会针对每一条序列，单独生成若干 query token。

query token 是根据什么生成的？

- 所有 NS token
- 当前序列的 mean pooling 表示

直观理解：

- NS token 表示全局背景
- 当前序列的平均表示，代表这条序列的大致内容
- 两者合在一起，模型生成一个 query，告诉自己：
  - “我接下来要从这条序列里读什么信息？”

这很像：

- 带着一个问题去读一段历史

#### 10.5 再展开一点：它到底在算什么

这一块最核心的思想是：

- 先不要急着直接让 query 去 attention
- 先根据“全局背景 + 当前序列概况”生成一个“阅读意图”

代码里，针对每一条序列 `seq_i`，都会做下面这件事：

```text
GlobalInfo_i = concat(NS tokens flatten, mean_pool(seq_i))
Q_i = FFN(GlobalInfo_i)
```

如果 `num_queries > 1`，那就不是 1 个 FFN，而是多套 FFN，
每套生成 1 个 query token。

---

#### 第一步：先看输入 shape

假设：

- `batch_size = 2`
- `d_model = 64`
- 一共有 `12` 个 NS token
- 一共有 `4` 条序列
- `num_queries = 1`

那么进入 `MultiSeqQueryGenerator` 之前：

- `ns_tokens: [2, 12, 64]`
- `seq_a_tokens: [2, 5, 64]`
- `seq_b_tokens: [2, 5, 64]`
- `seq_c_tokens: [2, 5, 64]`
- `seq_d_tokens: [2, 5, 64]`

---

#### 第二步：把 NS token 展平

`ns_tokens: [2, 12, 64]`

会先 reshape 成：

- `ns_flat: [2, 12 * 64] = [2, 768]`

意思是：

- 每个样本有 12 个静态 token
- 现在先把它们摊平成一个大向量

---

#### 第三步：对当前序列做 mean pooling

以 `seq_a` 为例。

假设某个样本的 `seq_a_tokens` 是 5 个 4 维 token。
为了方便看数值，我们先把维度缩小到 4 来举例：

```text
t1 = [1, 0, 2, 1]
t2 = [0, 1, 1, 1]
t3 = [1, 1, 0, 2]
t4 = [0, 0, 1, 1]
t5 = [2, 1, 1, 0]
```

如果这 5 个位置都有效，那么 mean pooling 就是按维度求平均：

```text
seq_mean
= (t1 + t2 + t3 + t4 + t5) / 5
= [4, 3, 5, 5] / 5
= [0.8, 0.6, 1.0, 1.0]
```

这就是：

- “这条序列的大致摘要”

如果真实模型里 `d_model = 64`，那这个摘要向量的 shape 就是：

- `[64]`

对于整个 batch：

- `seq_a_pooled: [2, 64]`

---

#### 第四步：把全局静态背景和当前序列摘要拼起来

现在有两部分：

- `ns_flat: [2, 768]`
- `seq_a_pooled: [2, 64]`

拼接后：

- `global_info_a: [2, 832]`

因为：

- `832 = 768 + 64 = (12 + 1) * 64`

这也是代码里 `global_info_dim = (num_ns + 1) * d_model` 的来源。

直观上，这个向量包含：

- 这个用户/商品的静态背景
- 这条序列整体在表达什么

---

#### 第五步：用 FFN 生成 query token

然后 `global_info_a` 会过一套小 MLP。

如果 `num_queries = 1`，那么对 `seq_a` 只生成 1 个 query：

- 输入：`[2, 832]`
- 输出：`[2, 64]`

再在 token 维上 `unsqueeze` 一下，就成了：

- `q_a: [2, 1, 64]`

这里的 `1` 才表示：

- 对 `seq_a` 只生成 1 个 query token

不是说 query 里面只有 1 个数字。

如果 `num_queries = 2`，那就会有两套 FFN，例如：

- `q_a_1: [2, 64]`
- `q_a_2: [2, 64]`

stack 之后就是：

- `q_a: [2, 2, 64]`

你可以把它理解成：

- 同一条序列，模型准备用 2 个不同的“问题”去读它

---

#### 第六步：四条序列各自生成自己的 query

因为这个项目有四路序列：

- `seq_a`
- `seq_b`
- `seq_c`
- `seq_d`

所以最后会得到：

- `q_a: [B, Nq, D]`
- `q_b: [B, Nq, D]`
- `q_c: [B, Nq, D]`
- `q_d: [B, Nq, D]`

如果：

- `B = 2`
- `Nq = 1`
- `D = 64`

那就是：

- `q_a: [2, 1, 64]`
- `q_b: [2, 1, 64]`
- `q_c: [2, 1, 64]`
- `q_d: [2, 1, 64]`

---

#### 为什么不直接用一个固定的 [CLS]？

因为这个项目希望：

- 不同序列有不同 query
- query 不是固定参数，而是跟当前样本内容相关

例如：

- 用户最近浏览序列里全是跑鞋
- 当前商品也是跑鞋

那生成出来的 query 可能更偏向“读取鞋类购买意图”。

如果另一个样本里：

- 用户最近一直在看手机

那 query 的方向就会不一样。

所以它不是一个死的 `[CLS]`，
而是一个：

- 由当前样本动态生成的“查询向量”

---

#### 一句话总结 10.5

`MultiSeqQueryGenerator` 做的事可以浓缩成一句话：

- 先看“我是谁 + 我这条历史大概在讲什么”，再生成一个 query，拿着这个 query 去后面的序列里读重点。

### 10.6 模块 4：MultiSeqHyFormerBlock

这是模型的主体模块，会堆叠多层。

每一层大致做三件事：

1. 每条序列先各自演化
2. query token 去读各自的序列
3. 所有 query token 和 NS token 再融合

#### 第一步：Sequence Evolution

每条序列的 token 先过自己的编码器。

如果是 `transformer`：

- 就是自注意力 + FFN

如果是 `swiglu`：

- 就是更轻量的 FFN 风格

如果是 `longer`：

- 对长序列做压缩处理

#### 第二步：Query Decoding

对于第 i 条序列：

- query token 作为 Q
- 这条序列 token 作为 K/V

也就是：

- query 去这条序列里读信息

这一步之后，query token 会变成：

- 吸收了当前序列关键信息的表示

#### 第三步：Token Fusion / RankMixer

再把：

- 所有序列读出来的 query token
- 以及 NS token

拼到一起做融合。

为什么要做这一步？

因为不同来源的信息之间是有关联的。

例如：

- 浏览历史表明你最近在看跑鞋
- 用户画像表明你是高消费客群
- 当前商品又是高价品牌跑鞋

这些信号单独看不够，需要一起融合。

### 10.7 模块 5：输出层

所有 block 跑完后：

- 取所有序列的 query token
- 拼接
- 展平
- 过一个投影层
- 再过分类头

最终得到：

- 一个 logit

这个 logit 经过 sigmoid 后，就可以理解成转化概率。

---

## 11. 用一个具体例子把前向传播走一遍

我们现在造一个非常简化的例子。

假设：

- batch size = 2
- 序列域有 4 条
- `d_model = 64`
- `num_queries = 1`

### 11.1 原始输入 shape

假设某个 batch 的 shape 如下：

- `user_int_feats: [2, 30]`
- `item_int_feats: [2, 10]`
- `user_dense_feats: [2, 918]`
- `seq_a: [2, 3, 5]`
- `seq_b: [2, 4, 5]`
- `seq_c: [2, 2, 5]`
- `seq_d: [2, 6, 5]`

解释：

- `seq_a` 在每个时间步有 3 个 side-info 字段
- `seq_b` 有 4 个
- `seq_c` 有 2 个
- `seq_d` 有 6 个
- 每条序列长度都截断到 5

### 11.2 先做静态特征 token 化

假设：

- 用户离散特征被分成 7 个 token
- 用户 dense 特征投影成 1 个 token
- 商品离散特征被分成 4 个 token

那么最终：

- `ns_tokens: [2, 12, 64]`

含义：

- 2 个样本
- 每个样本有 12 个静态 token
- 每个 token 是 64 维

### 11.3 再把每条序列变成 token 序列

经过 `_embed_seq_domain()` 后：

- `seq_a_tokens: [2, 5, 64]`
- `seq_b_tokens: [2, 5, 64]`
- `seq_c_tokens: [2, 5, 64]`
- `seq_d_tokens: [2, 5, 64]`

你看这里发生了最重要的变换：

- 原始是 `[B, S, L]`
- 现在变成 `[B, L, D]`

也就是：

- 每个时间步现在有一个统一维度的表示

### 11.4 为每条序列生成 query token

因为 `num_queries = 1`，所以每条序列生成 1 个 query：

- `q_a: [2, 1, 64]`
- `q_b: [2, 1, 64]`
- `q_c: [2, 1, 64]`
- `q_d: [2, 1, 64]`

你可以把它理解成：

- 每条序列都有一个“我要重点读什么”的查询向量

### 11.5 进入一个 HyFormer block

先让 4 条序列各自编码：

- 4 条序列各走自己的 sequence encoder

然后让 query 去读各自序列：

- `q_a` 去读 `seq_a_tokens`
- `q_b` 去读 `seq_b_tokens`
- `q_c` 去读 `seq_c_tokens`
- `q_d` 去读 `seq_d_tokens`

此时，query token 就带上了“从对应序列里读出来的信息”。

再把全部 query 和静态 token 拼起来：

- 4 个 query token
- 12 个 NS token
- 一共 16 个 token

于是：

- `combined: [2, 16, 64]`

再让 RankMixer 做融合。

### 11.6 输出预测

取更新后的所有 query：

- `[2, 4, 64]`

展平：

- `[2, 256]`

投影：

- `[2, 64]`

分类头：

- `[2, 1]`

这就是最终输出的 logit。

---

## 12. trainer 训练器在做什么

虽然类名叫 `PCVRHyFormerRankingTrainer`，但当前实现不是 listwise ranking，也不是 pairwise ranking。

它实际做的是：

- pointwise binary classification

也就是：

- 每个样本独立做一个二分类

### 12.1 一个训练 step 的流程

一个训练 step 大概是：

1. 把 batch 放到 GPU/CPU
2. 组装成 `ModelInput`
3. 跑 `model(model_input)`
4. 得到 logits
5. 算 loss
6. 反向传播
7. 梯度裁剪
8. 优化器更新

### 12.2 为什么有两个优化器

项目把参数分成两类：

- sparse params
  - 也就是 embedding 表

- dense params
  - 其余线性层、注意力层等参数

对应优化器：

- sparse -> `Adagrad`
- dense -> `AdamW`

这在推荐系统里很常见。

为什么？

因为 embedding 这种稀疏参数的更新方式和普通 dense 参数不一样，
很多工业推荐模型会单独给 embedding 设计优化策略。

### 12.3 验证时看什么指标

项目验证时主要看：

- AUC
- logloss

其中 early stopping 主要是看：

- AUC

因为在二分类预估里：

- AUC 能看排序区分能力
- logloss 能看概率质量

---

## 13. 这个项目里那些“奇怪的技巧”分别是干嘛的

如果你第一次看推荐模型，会觉得里面有些技巧很怪。其实它们大多是为了解决工业场景中的真实问题。

### 13.1 `emb_skip_threshold`

作用：

- 某个离散特征词表太大时，直接不给它建 embedding
- 运行时用零向量代替

为什么需要这个？

因为推荐系统里有些 id 特征基数巨大，比如：

- 商品 id
- 用户 id
- 内容 id

如果无脑给所有特征建 embedding：

- 显存会爆
- 训练成本会非常高

这个策略的本质是：

- 用建模能力换资源可控性

### 13.2 `seq_id_threshold`

作用：

- 序列中词表特别大的特征，会被认为是 id 型特征
- 对这些特征施加更强的 dropout

为什么？

因为高基数 id 特征特别容易记忆训练样本，从而过拟合。

### 13.3 高基数 embedding 重新初始化

项目支持在若干 epoch 后，把高基数 embedding 重置。

为什么要这么做？

因为在 CTR/CVR 场景里，高基数 embedding 容易在多轮训练里“记死”训练集，泛化变差。

这种做法本质上是一种：

- 针对 embedding 过拟合的特殊 regularization trick

### 13.4 time bucket embedding

作用：

- 把“离现在多久”这件事注入到序列表示中

为什么它重要？

因为推荐系统里时效性很强：

- 刚看过的行为
- 和很久以前看过的行为

含义通常不一样。

### 13.5 RoPE

作用：

- 给 attention 注入相对位置信息

为什么有帮助？

因为顺序很重要：

- 最近一次点击
- 和更早一次点击

在序列建模里语义不同。

---

## 14. 初学者该怎么学这个项目

如果你想“真正看懂”，强烈建议不要按文件从头读到尾。

最有效的路线是分 4 个阶段。

### 阶段 1：只看主流程

只看这些点：

- `train.py -> main`
- `dataset.py -> get_pcvr_data`
- `trainer.py -> _make_model_input`
- `model.py -> PCVRHyFormer.forward`

目标：

- 你能用自己的话说出“从 parquet 到 logits”这条路径

这一步不要追求细节，先建立全局感。

### 阶段 2：只啃数据层

重点看：

- `FeatureSchema`
- `_load_schema()`
- `_convert_batch()`

你必须搞清楚：

- `user_int_feats` 是怎么拼出来的
- `item_int_feats` 是怎么拼出来的
- `seq_a` 为什么是 `[B, S, L]`
- `seq_a_time_bucket` 是怎么来的

这一步如果没搞懂，后面模型基本看不明白。

### 阶段 3：按模块看模型

推荐顺序：

1. `GroupNSTokenizer` / `RankMixerNSTokenizer`
2. `_embed_seq_domain()`
3. `MultiSeqQueryGenerator`
4. `MultiSeqHyFormerBlock`
5. `PCVRHyFormer.forward()`

每次只问两个问题：

- 输入 shape 是什么
- 输出 shape 是什么

很多复杂模型其实就是 shape 变化叠出来的。

### 阶段 4：最后看训练技巧

最后再去看：

- dual optimizer
- early stopping
- focal loss
- embedding reinit
- longer encoder
- RoPE

这些是进阶理解，不是主干。

### 14.1 我最建议你做的 3 个练习

#### 练习 1：手画数据流图

把这些节点画出来：

- parquet 数据
- schema.json
- dataset
- batch dict
- ModelInput
- ns_tokens
- seq_tokens
- q_tokens
- blocks
- classifier

#### 练习 2：手工推出 shape

假设一个 batch：

```text
user_int_feats: [32, 30]
item_int_feats: [32, 10]
user_dense_feats: [32, 918]
seq_a: [32, 3, 256]
seq_b: [32, 4, 256]
seq_c: [32, 2, 512]
seq_d: [32, 6, 512]
```

你自己往后推每一步的 shape。  
这个练习会让你理解速度非常快。

#### 练习 3：用自己的话解释 `_embed_seq_domain()`

如果你已经可以解释这一个函数，那你对这个项目的理解就已经过了第一道坎。

---

## 15. 这个项目目前有哪些明显的可改进方向

这一节是站在“后续可以继续研究和迭代”的角度写的。  
我会按下面这个格式来讲：

- 当前项目的现状
- 为什么可能不够
- 可以怎么改
- 参考论文

注意：

- 下面这些不是说当前项目不能用
- 而是说，如果你想把它做得更强、更贴近工业推荐研究，这些方向值得优先考虑

### 15.1 改进方向一：把 PCVR 做成更合理的多任务建模，而不是只做单一 post-click 二分类

#### 当前项目现状

当前项目主要做的是：

- 单任务二分类
- 标签来自 `label_type == 2`

这意味着模型只在学：

- 点击后会不会转化

#### 为什么这可能不够

在真实推荐/广告场景里，CVR/PCVR 数据通常比 CTR 稀疏很多。

问题是：

- 转化样本少
- 学习信号弱
- 容易过拟合

而且从业务逻辑上，转化通常依赖于点击：

- 不点击，通常就不会发生点击后转化

所以很多工业模型不会只孤立建模 CVR，
而会把点击和转化一起建模。

#### 可以怎么改

可以考虑：

1. 增加 CTR 任务
2. 联合建模 CTR 和 CVR
3. 进一步做 CTCVR 或者 post-click multi-task

最直接的思路有两类：

- ESMM 风格
  - 解决样本选择偏差和数据稀疏问题

- MMoE / PLE 风格
  - 多任务共享底层，再分任务塔

#### 为什么值得改

因为这样通常能带来：

- 更稳定的训练
- 更好的泛化
- 更符合业务因果路径

#### 可参考论文

- ESMM: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate
- MMoE: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
- PLE: Progressive Layered Extraction for Multi-Task Learning in Personalized Recommendations

你可以这样理解：

- ESMM 更贴近 CVR/CTCVR 特殊问题
- MMoE/PLE 更偏通用多任务结构升级

### 15.2 改进方向二：让序列建模更“目标感知”，而不是只靠统一 query 去读历史

#### 当前项目现状

当前项目已经有 query token 机制，这很好。  
但它的 query 主要来自：

- NS token
- 当前序列的 mean pooling

它还没有非常显式地使用：

- 当前候选商品本身

来决定“读历史时关注哪部分”。

#### 为什么这可能不够

在推荐系统里，一个很经典的问题是：

- 用户历史很长
- 但不是所有历史都和当前候选商品相关

例如：

- 用户之前看过手机、电脑、跑鞋、零食
- 当前候选商品是一双跑鞋

那最相关的，通常是和跑鞋、运动、鞋类相关的历史，而不是全部历史平均重要。

#### 可以怎么改

把“当前候选 item 表示”更显式地注入到序列注意力里。

例如：

- 让当前 item token 作为 query
- 对历史序列做 target-aware attention
- 或者在 query generator 中加入当前 item 的显式条件

#### 为什么值得改

因为这会让模型从：

- “泛化地读历史”

变成：

- “围绕当前候选商品，有针对性地读历史”

这在推荐场景里通常非常重要。

#### 可参考论文

- DIN: Deep Interest Network for Click-Through Rate Prediction
- DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction

初学者可以这样理解：

- DIN 强调“当前候选商品不同，读历史的关注点也应该不同”
- DIEN 在 DIN 基础上更强调“兴趣是动态演化的”

### 15.3 改进方向三：把时间建模从“简单分桶”升级为“注意力级别的时间间隔建模”

#### 当前项目现状

现在项目里已经有时间建模：

- 通过 `time_bucket` 做 embedding

这已经比完全不用时间强很多。

#### 为什么这可能还不够

因为“时间差”现在只是：

- 先离散成 bucket
- 再加到 token 上

这是一种比较朴素的做法。

但很多时候，时间信息不应该只是一个附加 embedding，
而应该直接影响注意力权重。

例如：

- 刚发生的行为应该更容易被关注
- 不同时间间隔下，行为相关性模式不同

#### 可以怎么改

考虑把时间间隔直接纳入 self-attention / cross-attention 的打分里。

例如：

- 用相对时间间隔偏置
- 用 time-aware self-attention
- 用 interval-aware attention

#### 为什么值得改

因为在行为序列推荐里，时间信息往往是强信号。

简单说：

- “最近发生” 和 “很久以前发生”

不只是两个不同 embedding，它们还会改变“该不该关注、关注多少”。

#### 可参考论文

- TiSASRec: Time Interval Aware Self-Attention for Sequential Recommendation

这篇论文很适合你理解：

- 时间不是附属信息
- 时间可以直接参与序列注意力计算

### 15.4 改进方向四：增强静态特征的显式高阶交叉，而不是主要依赖 Transformer 隐式学习

#### 当前项目现状

当前项目对静态特征做了 token 化，然后通过后续模块融合。  
这很灵活，但也有一个问题：

- 有些结构化特征交叉，其实用显式 cross 网络会更高效

#### 为什么这可能不够

推荐系统里的结构化特征常常有很强的组合关系：

- 用户年龄段 x 商品类目
- 城市等级 x 价格带
- 用户消费水平 x 品牌档次

这些关系虽然 Transformer 可能也能学到，
但不一定是最参数高效、最稳定的方式。

#### 可以怎么改

在当前模型的静态分支后面，加一层或几层显式交叉模块，例如：

- DCN V2 风格 cross network

然后把这一路和现有 HyFormer 输出再融合。

#### 为什么值得改

这样做的好处通常是：

- 对表格型特征更友好
- 学到的交叉更显式
- 参数利用率更高

#### 可参考论文

- DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

如果你以后想做“更强的静态特征建模”，这篇论文很值得读。

### 15.5 改进方向五：在序列学习上加入自监督预训练或对比学习

#### 当前项目现状

当前训练方式是纯监督：

- 给定标签
- 直接做 PCVR 二分类

#### 为什么这可能不够

因为用户行为序列里其实有很多“无标签结构信息”没有被充分利用。

比如：

- 哪些行为经常相邻出现
- 哪些商品兴趣具有替代关系
- 哪些行为顺序代表强意图

如果只靠最终转化标签来学，可能比较浪费序列信息。

#### 可以怎么改

可以考虑在主任务之外加入：

- masked behavior modeling
- sequence contrastive learning
- next behavior prediction

先做预训练，再做下游 PCVR 微调，或者联合训练。

#### 为什么值得改

这种方法通常能：

- 让序列表示更稳
- 更充分利用无标签行为数据
- 在标签稀疏时更有帮助

#### 可参考论文

- CL4SRec: Contrastive Learning for Sequential Recommendation

如果你想往“序列表示学习”方向扩展，这是一个比较自然的切入点。

### 15.6 改进方向六：把训练和验证切分方式做得更贴近真实线上场景

#### 当前项目现状

现在是：

- 按 parquet row group 切 train/valid

#### 为什么这可能不够

如果数据有时间性，或者 row group 本身有特殊排序，
那么当前验证集不一定能真实反映线上效果。

尤其在推荐系统里，分布漂移很常见：

- 用户兴趣会变
- 商品集合会变
- 流量入口会变

#### 可以怎么改

优先考虑：

- 时间切分验证
- 按日期滚动验证
- 更接近线上回放的评估协议

#### 为什么值得改

因为推荐系统里一个非常重要的问题是：

- 离线指标看着很好
- 线上效果不一定跟着好

改进验证协议，经常比盲目加模型更重要。

#### 这一点的论文参考

这类问题更多是工业实验设计和评测协议问题，不一定对应某一篇单独的结构论文。  
如果你后面进入更深入的推荐系统工程学习，会经常看到：

- temporal split
- online-offline gap
- counterfactual evaluation

这些关键词。

---

## 16. 这些改进方向里，初学者最值得先做哪几个

如果你现在还是新手，我建议优先级如下：

1. 多任务学习改造
2. 目标感知历史建模
3. 更强的时间建模
4. 改进验证切分方式
5. 显式静态特征交叉
6. 自监督序列预训练

为什么这么排序？

- 多任务和目标感知，是推荐系统里收益最常见也最直观的方向
- 时间建模对行为序列也非常重要
- 评测协议会直接影响你对模型好坏的判断
- 静态交叉和自监督更适合在你基础更稳以后再深入

---

## 17. 推荐你接下来怎么继续学

如果你愿意继续按这个项目深入，我建议路线是：

### 第一步

先只看：

- `train.py -> main`
- `dataset.py -> _load_schema()`
- `dataset.py -> _convert_batch()`

目标：

- 彻底搞懂一个 batch 是怎么来的

### 第二步

再看：

- `model.py -> GroupNSTokenizer`
- `model.py -> RankMixerNSTokenizer`
- `model.py -> _embed_seq_domain()`

目标：

- 搞懂静态特征和序列特征分别如何 token 化

### 第三步

再看：

- `MultiSeqQueryGenerator`
- `MultiSeqHyFormerBlock`
- `PCVRHyFormer.forward()`

目标：

- 把整个前向传播串起来

### 第四步

最后看：

- `trainer.py`
- `utils.py`

目标：

- 搞懂 loss、优化器、验证和 early stopping

---

## 18. 一句话总结整个项目

这个项目本质上是一个面向推荐/广告场景的多路行为序列预测模型：

- 用 `schema.json` 解释原始 parquet 特征
- 把静态特征压成 NS token
- 把每条行为序列变成一串时间步 token
- 用 query 去不同序列中读取和当前样本相关的信息
- 再把这些信息融合起来，预测是否转化

如果你已经能用自己的话把上面这句话讲明白，
说明你已经真的理解了这个项目的主干。

---

## 19. 推荐论文清单

下面这些论文是本节“改进方向”里提到的代表性工作。  
你不用现在全读，但建议把名字先记住。

- ESMM: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate
- MMoE: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
- PLE: Progressive Layered Extraction for Multi-Task Learning in Personalized Recommendations
- DIN: Deep Interest Network for Click-Through Rate Prediction
- DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction
- TiSASRec: Time Interval Aware Self-Attention for Sequential Recommendation
- DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
- CL4SRec: Contrastive Learning for Sequential Recommendation

你可以按这个顺序挑着看：

1. DIN
2. DIEN
3. ESMM
4. TiSASRec
5. DCN V2
6. MMoE / PLE
7. CL4SRec

这个顺序对初学者更友好一些。
