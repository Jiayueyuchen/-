import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import itertools

# 数据读取
user_feature = pd.read_csv('./data/user.txt', encoding='utf-8', sep='\t')
item_feature = pd.read_csv('./data/weather.txt', encoding='utf-8', sep='\t')
rating = pd.read_csv('./data/rating.txt', encoding='utf-8', sep='\t')

# 创建用户-物品评分矩阵
user_item_matrix = rating.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# 使用KNN算法计算用户之间的相似度
knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn.fit(user_item_matrix.values)

# 总样本数
N = len(rating)


# 推荐Top-K (使用KNN来找到相似用户，并推荐这些相似用户喜欢的物品)
def recommend_top_k(user_id, top_k=3):
    # 获取该用户的评分向量
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # 查找与该用户最相似的 top_k 个用户
    distances, indices = knn.kneighbors(user_vector, n_neighbors=top_k + 1)  # +1 是因为包含了用户本身
    similar_users = indices.flatten()[1:]  # 排除自己

    # 获取这些相似用户的物品评分
    similar_users_ratings = user_item_matrix.iloc[similar_users]

    # 对于每个物品，计算这些相似用户的平均评分
    avg_ratings = similar_users_ratings.mean(axis=0)

    # 推荐评分最高的 top_k 个物品
    top_k_items = avg_ratings.argsort()[-top_k:][::-1]
    return user_item_matrix.columns[top_k_items]

# 获取该用户的正样本：评分大于等于3的物品
def get_positive_samples(user_id, rating_data):
    # 筛选出该用户的评分数据
    user_ratings = rating_data[rating_data['user_id'] == user_id]
    # 筛选出评分大于等于3的物品
    positive_samples = set(user_ratings[user_ratings['rating'] >= 3]['item_id'])
    return positive_samples

# 计算准确率
def accuracy(user_id, top_k_items, rating_data, k=3):
    # 获取该用户的正样本
    positive_samples = get_positive_samples(user_id, rating_data)
    # 推荐的物品
    recommended_items = set(top_k_items)

    # 计算正样本与推荐物品的交集
    true_positive = len(positive_samples & recommended_items)
    false_positive = k - true_positive
    false_negative = len(positive_samples) - true_positive
    true_negative = N - true_positive - false_positive - false_negative

    accuracy_score = (true_positive + true_negative) / N
    return accuracy_score

# 计算精确度 (Precision)
def precision(user_id, top_k_items, rating_data, k=3):
    # 获取该用户的正样本
    positive_samples = get_positive_samples(user_id, rating_data)
    # 推荐的物品
    recommended_items = set(top_k_items)

    # 计算 Precision
    true_positive = len(positive_samples & recommended_items)
    false_positive = k - true_positive
    precision_score = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    return precision_score

# 计算召回率 (Recall)
def recall(user_id, top_k_items, rating_data, k=3):
    # 获取该用户的正样本
    positive_samples = get_positive_samples(user_id, rating_data)
    # 推荐的物品
    recommended_items = set(top_k_items)

    # 计算 Recall
    true_positive = len(positive_samples & recommended_items)
    false_negative = len(positive_samples) - true_positive
    recall_score = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    return recall_score

# 计算 F1 分数
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# 记录查询结果
results = []
max_queries = 5  # 限制最大查询次数

while len(results) < max_queries:
    user_input = input("请输入用户id (输入 'quit' 退出): ")

    # 检查输入是否为 'quit'
    if user_input.lower() == 'quit':
        print("退出程序")
        break  # 退出循环

    # 调试信息：打印用户输入
    print(f"用户输入: {user_input}")

    try:
        # 尝试将输入转换为整数
        user_id = int(user_input)
        top_k_items = recommend_top_k(user_id, 3).tolist()

        # 计算准确率、Precision、Recall 和 F1
        acc = accuracy(user_id, top_k_items, rating, k=3)
        prec = precision(user_id, top_k_items, rating, k=3)
        rec = recall(user_id, top_k_items, rating, k=3)
        f1 = f1_score(prec, rec)
        # 输出推荐结果和评估指标
        print(f"推荐物品: {top_k_items}")

        print(f"用户{user_id}的Top-3推荐准确率: {acc:.4f}")
        print(f"用户{user_id}的推荐精确度: {prec:.4f}")
        print(f"用户{user_id}的推荐召回率: {rec:.4f}")
        print(f"用户{user_id}的推荐F1值: {f1:.4f}")

        # 保存当前用户的查询结果
        results.append((user_id, top_k_items, acc, prec, rec, f1))

    except ValueError:
        print(f"输入无效：'{user_input}'，请输入一个有效的用户ID或输入 'quit' 退出。")


columns = ['user_id', 'recommended_item', 'accuracy', 'precision', 'recall', 'f1']
results_df = pd.DataFrame(results, columns=columns)

print(results_df)