import torch
import torch_explain as te
# from torch_explain import datasets
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
from torchvision.models import resnet50,densenet169,vgg16
from torchvision import transforms,datasets
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.utils as torch_utils
import wandb
from models import Neural_Concat_Model,DenseNetModel,VitModel,Neural_Concat_vitbackbone_Model
from datetime import datetime
import os
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
from dataset.ddi_concept_dataset import DDI_Dataset,ImageFolderWithPaths
from tqdm import trange
from models import ResNetModel,DenseNetModel
from model.unet_model import UNet
from vit_pytorch import ViT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

parser = argparse.ArgumentParser(description='Settings for creating model')
parser.add_argument("--concept_path",type=str,default='D:\\project\\XAI4Medical\\Label-free-CBM-main\\data\\concept_sets\\covid19_ct_designed.txt')
parser.add_argument("--dataset",type=str,default='covid19-ct')
parser.add_argument("--image_dir",type=str,default='D:\\project\\XAI4Medical\\pytorch_explain\\DDI')
parser.add_argument("--embedding_size",type=int,default=8)
parser.add_argument("--batchsize",type=int,default=64)
parser.add_argument("--n_classes",type=int,default=2)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--val_epoch_per",type=int,default=2)
parser.add_argument("--feature_size",type=int,default=2048)
parser.add_argument("--backbone",type=str,default='DenseNet')
parser.add_argument("--savedir",type=str,default=f'D:\\project\\XAI4Medical\\pytorch_explain\\results\\{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
parser.add_argument("--wandb",default=True,action='store_true', help="record information on wandb")
args = parser.parse_args()
def lambda_rule(epoch):
    return 0.001 / (1 + epoch)
if args.wandb==True:
    wandb.init(project="neural", name=f"backbone_{args.backbone}epoch_{args.epochs};embeddingsize_{args.embedding_size};lr=0.1",group='baseline_model')
# x, c, y = datasets.xor(500)
# x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)
concepts = utils.get_concepts(args.concept_path)
concepts_num=len(concepts)
train_data=data_utils.get_data(f'{args.dataset}_train')
test_data=data_utils.get_data(f'{args.dataset}_val')
train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
if args.backbone=='RN50':
    backbone = resnet50(pretrained=True)

    # backbone = nn.Sequential(*list(backbone.children())[:-1])
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model=ResNetModel(backbone,task_predictor)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
if args.backbone=='vgg':
    backbone = vgg16(pretrained=True)
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
       backbone,
       task_predictor 
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
if args.backbone=='DenseNet':
    feature_size = 1000
    backbone = densenet169(pretrained=True)
    
    task_predictor = torch.nn.Sequential(
         torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
       backbone,
       task_predictor 
    )
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

if args.backbone=='unet':
    backbone = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

    
    task_predictor = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(2, 16),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(16, 2),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
       backbone,
       task_predictor 
    )
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
if args.backbone=='vit':
    # Load model directly
    backbone = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

    feature_size=1000
    task_predictor = torch.nn.Sequential(
         torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
       backbone,
       task_predictor 
    )
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
model.train()
loss_task = torch.nn.CrossEntropyLoss()
# scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
best_val_accuracy=0.0
for epoch in range(args.epochs):
    train_y_true = torch.tensor([]).to(device)
    train_y_pred = torch.tensor([]).to(device)

    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch'):
        optimizer.zero_grad()  
        y_pred=model(images.to(device))    
        loss=loss_task(y_pred, F.one_hot(labels.long().ravel()).float().to(device))
        # train_y_true = torch.cat((train_y_true, labels.view(labels.size(0), -1).to(device)), dim=0)
        train_y_true = torch.cat((train_y_true, F.one_hot(labels.long().ravel()).float().to(device)))
        train_y_pred = torch.cat((train_y_pred, (y_pred.cpu() > 0.5).to(device)), dim=0)
        # train_concepts_true = torch.cat((train_concepts_true, concepts_batch.to(device)), dim=0)
        # train_concepts_pred = torch.cat((train_concepts_pred, (c_pred.cpu() > 0.5).to(device)), dim=0)
        if args.wandb==True:
            wandb.log({'loss':loss,'lr':optimizer.param_groups[0]["lr"]})
        
        print(f' loss:{loss}')
        loss.backward()
        
        optimizer.step()
    train_task_accuracy = accuracy_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy())
    # train_concept_accuracy = accuracy_score(train_concepts_true.cpu().numpy(), train_concepts_pred.cpu().numpy())
    print( f'train_task_accuracy:{train_task_accuracy}')          
    # local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
    # global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
    
    # scheduler.step(loss)
    # scheduler.step()
    if args.wandb==True:
        wandb.log({'train_task_accuracy':train_task_accuracy})
    all_y_true = []
    all_y_pred = []
    concept_true=[]
    all_y_concept_pred=[]
    concept_num_total=[]
    if epoch % args.val_epoch_per==0 :
        with torch.no_grad():  # 在评估时不需要计算梯度
            for batch in tqdm(test_loader, desc='val'):
                images, labels=batch
                optimizer.zero_grad()      
                y_pred=model(images.to(device))    

        # train_y_true = torch.cat((train_y_true, labels.view(labels.size(0), -1).to(device)), dim=0)
                train_y_true = torch.cat((train_y_true, F.one_hot(labels.long().ravel()).float().to(device)))
                train_y_pred = torch.cat((train_y_pred, (y_pred.cpu() > 0.5).to(device)), dim=0)

                # all_y_true.extend(labels.view(labels.size(0), -1).cpu().numpy())
                all_y_true.extend(F.one_hot(labels.long().ravel()).float().cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
                
        model.train()
        # 计算整体的准确度
        task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))
        task_f1 = f1_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算Precision
        task_precision = precision_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算Recall
        task_recall = recall_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算ROC和AUC
        # fpr, tpr, _ = roc_curve(np.array(all_y_true), np.array(all_y_pred))
        task_auc = roc_auc_score(np.array(all_y_true), np.array(all_y_pred))

        if args.wandb==True:
            wandb.log({'task_accuracy':task_accuracy,'task_f1':task_f1,'task_precision':task_precision,'task_recall':task_recall,'task_auc':task_auc})
        
        if task_accuracy > best_val_accuracy:
                best_val_accuracy = task_accuracy
                if not os.path.exists(args.savedir):
                    os.mkdir(args.savedir)
                torch.save(model.state_dict(), os.path.join(args.savedir, 'best_model.pth'))
        print('best_val_accuracy:',best_val_accuracy,'task_accuracy:',task_accuracy,'task_f1:',task_f1,'task_precision:',task_precision,'task_recall:',task_recall,'task_auc:',task_auc)

