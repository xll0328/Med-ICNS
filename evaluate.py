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
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.utils as torch_utils
import wandb
from models import Neural_Concat_Model,DenseNetModel,VitModel,Neural_Concat_vitbackbone_Model
from datetime import datetime
import os
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
# 其他导入语句...

# 参数设置...
args = argparse.Namespace(
    concept_path='D:\\project\\XAI4Medical\\Label-free-CBM-main\\data\\concept_sets\\covid19_ct_designed.txt',
    dataset='covid19-ct',
    embedding_size=8,
    batchsize=32,
    n_classes=2,
    epochs=100,
    backbone='DenseNet',
    savedir=f'D:\\project\\XAI4Medical\\pytorch_explain\\results\\2023-12-22_21-29-09',
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
    backbone = resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 128),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(128, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 1),
        # nn.Sigmoid()
    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
if args.backbone=='DenseNet':
    feature_size = 81536
    backbone = densenet169(pretrained=True)
    # backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 128),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(128, concepts_num, args.embedding_size),
    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    # num_features = densenet_model.classifier.in_features
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 1024),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(1024, 1),
        nn.Sigmoid()
    )
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
# 加载最佳模型权重
# best_model_path = os.path.join(args.savedir, 'best_model_complete.pth')

# model=torch.load(best_model_path)
best_model_path = os.path.join(args.savedir, 'best_model.pth')

model.load_state_dict(torch.load(best_model_path))
model.to(device)

# 设置模型为评估模式
model.eval()

all_y_true = []
all_y_pred = []
all_concepts_true = []
all_concepts_pred = []

with torch.no_grad():  # 在评估时不需要计算梯度
    for images, labels in tqdm(test_loader, desc='test'):
        concepts_batch = torch.tensor(utils.labeltoconcepts(labels, concepts)).to(device)
        feature, y_pred, y_pred_neural, c_emb, c_pred = model(images.to(device))

        local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
        global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')

        # 将真实标签和预测结果累积起来
        all_y_true.extend(labels.view(labels.size(0), -1).cpu().numpy())
        all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
        all_concepts_true.extend(concepts_batch.cpu().numpy())
        all_concepts_pred.extend(c_pred.cpu().numpy() > 0.5)

# 计算整体的准确度
task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))
concept_accuracy = accuracy_score(np.array(all_concepts_true), np.array(all_concepts_pred))

# 写入文件
result_file_path = os.path.join(args.savedir, 'evaluation_results.txt')
with open(result_file_path, 'w') as result_file:
    result_file.write(f'Task Accuracy: {task_accuracy}\n')
    result_file.write(f'Concept Accuracy: {concept_accuracy}\n')

print(f'Results written to {result_file_path}')
