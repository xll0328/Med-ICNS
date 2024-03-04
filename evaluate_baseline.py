import torch
import torch_explain as te
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptReasoningLayer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from torch_explain.nn.concepts import ConceptReasoningLayer
import torch.nn.functional as F
import utils
import argparse
import data_utils
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50,densenet169
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.utils as torch_utils
import wandb
from models import ResNetModel,DenseNetModel
from datetime import datetime
import os
from torch.optim.lr_scheduler import LambdaLR

# 参数设置...
args = argparse.Namespace(
    concept_path='D:\\project\\XAI4Medical\\Label-free-CBM-main\\data\\concept_sets\\covid19_ct_designed.txt',
    dataset='covid19-ct',
    embedding_size=8,
    batchsize=32,
    n_classes=2,
    epochs=100,
    backbone='DenseNet',
    savedir=f'D:\\project\\XAI4Medical\\pytorch_explain\\baseline_results\\2023-12-22_17-34-51',
    wandb=True
)
concepts = utils.get_concepts(args.concept_path)
concepts_num=len(concepts)
test_data=data_utils.get_data(f'{args.dataset}_test')
test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(42)
if args.backbone=='RN50':
    feature_size=2048
    resnet_model = resnet50(pretrained=True)

    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 1),
        # nn.Sigmoid()
    )
    model=ResNetModel(resnet_model,task_predictor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.to(device)
if args.backbone=='DenseNet':
    densenet_model = densenet169(pretrained=True)
    # densenet_model = torch.nn.Sequential(*list(densenet_model.children())[:-1])
    densenet_model = nn.Sequential(*list(densenet_model.children())[:-1])
    # num_features = densenet_model.classifier.in_features
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(81536, 1024),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(1024, 1),
        nn.Sigmoid()
    )
    model =DenseNetModel(densenet_model,task_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
# 加载最佳模型权重
best_model_path = os.path.join(args.savedir, 'best_model.pth')

model.load_state_dict(torch.load(best_model_path))
model.to(device)

# 设置模型为评估模式
model.eval()

all_y_true = []
all_y_pred = []

with torch.no_grad():  # 在评估时不需要计算梯度
    for images, labels in tqdm(test_loader, desc='test'):
        concepts_batch = torch.tensor(utils.labeltoconcepts(labels, concepts)).to(device)
        y_pred= model(images.to(device))

        # 将真实标签和预测结果累积起来
        y_pred=model(images.to(device))
            # 将真实标签和预测结果累积起来
        all_y_true.extend(labels.view(labels.size(0), -1).cpu().numpy())
        all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
    # 计算整体的准确度
task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))

# 写入文件
result_file_path = os.path.join(args.savedir, 'evaluation_results.txt')
with open(result_file_path, 'w') as result_file:
    result_file.write(f'Task Accuracy: {task_accuracy}\n')
    

print(f'Results written to {result_file_path}')