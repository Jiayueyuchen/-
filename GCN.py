import pandas as pd
import numpy as np
import time
from utils import fix_seed_torch, draw_loss_pic
import argparse
from model import GCN
from Logger import Logger
from mydataset import MyDataset
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import sys

# 固定随机数种子
fix_seed_torch(seed=2021)

# 设置训练的超参数
parser = argparse.ArgumentParser()
parser.add_argument('--gcn_layers', type=int, default=2, help='the number of gcn layers')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--embedSize', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
args = parser.parse_args()

# 设备是否支持cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device

# 读取用户特征、天气特征、评分
user_feature = pd.read_csv('./data/user.txt', encoding='utf-8', sep='\t')
item_feature = pd.read_csv('./data/weather.txt', encoding='utf-8', sep='\t')
rating = pd.read_csv('./data/rating.txt', encoding='utf-8', sep='\t')

# 构建数据集
dataset = MyDataset(rating)
trainLen = int(args.ratio * len(dataset))
train, test = random_split(dataset, [trainLen, len(dataset) - trainLen])
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=len(test))

# 记录训练的超参数
start_time = '{}'.format(time.strftime("%m-%d-%H-%M", time.localtime()))
logger = Logger('./log/log-{}.txt'.format(start_time))
logger.info(' '.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

# 定义模型
model = GCN(args, user_feature, item_feature, rating)
model.to(device)

# 定义优化器
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

# 定义损失函数
loss_function = MSELoss()
train_result = []
test_result = []

# 最好的epoch
best_loss = sys.float_info.max

# 训练
for i in range(args.n_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        prediction = model(batch[0].to(device), batch[1].to(device))
        train_loss = torch.sqrt(loss_function(batch[2].float().to(device), prediction))
        train_loss.backward()
        optimizer.step()

    train_result.append(train_loss.item())

    model.eval()
    for data in test_loader:
        prediction = model(data[0].to(device), data[1].to(device))
        test_loss = torch.sqrt(loss_function(data[2].float().to(device), prediction))
        test_loss = test_loss.item()
        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './model/bestModeParms-{}.pth'.format(start_time))
        test_result.append(test_loss)
        logger.info("Epoch{:d}:trainLoss{:.4f},testLoss{:.4f}".format(i, train_loss, test_loss))

# 画图
draw_loss_pic(train_result, test_result)

# 加载最佳模型
model.load_state_dict(torch.load('./model/bestModeParms-{}.pth'.format(start_time)))
model.eval()

# 评估指标：计算精确度、召回率、F1
def get_positive_samples(user_id, rating_data):
    user_ratings = rating_data[rating_data['user_id'] == user_id]
    positive_samples = set(user_ratings[user_ratings['rating'] >= 3]['item_id'])
    return positive_samples

# 计算准确率
def accuracy(user_id, top_k_items, rating_data, k=3):
    positive_samples = get_positive_samples(user_id, rating_data)
    recommended_items = set(top_k_items)
    true_positive = len(positive_samples & recommended_items)
    false_positive = k - true_positive
    false_negative = len(positive_samples) - true_positive
    true_negative = len(rating_data) - true_positive - false_positive - false_negative
    accuracy_score = (true_positive + true_negative) / len(rating_data)
    return accuracy_score

# 计算精确度 (Precision)
def precision(user_id, top_k_items, rating_data, k=3):
    positive_samples = get_positive_samples(user_id, rating_data)
    recommended_items = set(top_k_items)
    true_positive = len(positive_samples & recommended_items)
    false_positive = k - true_positive
    precision_score = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    return precision_score

# 计算召回率 (Recall)
def recall(user_id, top_k_items, rating_data, k=3):
    positive_samples = get_positive_samples(user_id, rating_data)
    recommended_items = set(top_k_items)
    true_positive = len(positive_samples & recommended_items)
    false_negative = len(positive_samples) - true_positive
    recall_score = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    return recall_score

# 计算 F1 分数
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 推荐Top-K并计算评估指标
while True:
    user_id = input("请输入用户id (输入 'quit' 退出):")
    if user_id == 'quit':
        break

    user_tensor = torch.tensor([int(user_id)] * model.num_item).to(device)  # 用户ID转换为张量
    item_tensor = torch.arange(model.num_item).to(device)  # 创建一个物品张量

    predictions = model(user_tensor, item_tensor)
    top_k_items = torch.topk(predictions, k=3)[1]

    # 计算评估指标
    top_k_items_list = top_k_items.cpu().numpy().tolist()
    acc = accuracy(int(user_id), top_k_items_list, rating, k=3)
    prec = precision(int(user_id), top_k_items_list, rating, k=3)
    rec = recall(int(user_id), top_k_items_list, rating, k=3)
    f1 = f1_score(prec, rec)

    print("推荐项目:")
    for item in top_k_items:
        item = item.item()
        print(item_feature.loc[item]['id'])

    # 打印评估结果
    print(f"用户{user_id}的Top-3推荐准确率: {acc:.4f}")
    print(f"用户{user_id}的推荐精确度: {prec:.4f}")
    print(f"用户{user_id}的推荐召回率: {rec:.4f}")
    print(f"用户{user_id}的推荐F1值: {f1:.4f}")
