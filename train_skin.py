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
from model.unet_model import UNet
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
parser.add_argument("--backbone",type=str,default='RN50')
parser.add_argument("--savedir",type=str,default=f'D:\\project\\XAI4Medical\\pytorch_explain\\results\\{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
parser.add_argument("--wandb",default=True,action='store_true', help="record information on wandb")
args = parser.parse_args()

if args.wandb==True:
    wandb.init(project="neural", name=f"backbone{args.backbone};epoch_{args.epochs};embeddingsize_{args.embedding_size};lr=0.01;concept_160;learning_schedule")
# x, c, y = datasets.xor(500)
# x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)
device='cuda'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(42)
# concepts=['Vesicle','Papule','Macule','Plaque','Abscess','Pustule','Bulla','Patch','Nodule','Ulcer',
#           'Crust','Erosion','Excoriation','Atrophy','Exudate','Purpura/Petechiae','Fissure','Induration',
#           'Xerosis','Telangiectasia','Scale','Scar','Friable','Sclerosis','Pedunculated','Exophytic/Fungating',
#           'Warty/Papillomatous','Dome-shaped','Flat topped','Brown(Hyperpigmentation)','Translucent','White(Hypopigmentation)',
#           'Purple','Yellow','Black','Erythema','Comedo','Lichenification','Blue','Umbilicated','Poikiloderma','Salmon',
#           'Wheal','Acuminate','Burrow','Gray','Pigmented','Cyst'
# ]
concepts=['Vesicle','Papule','Macule','Plaque','Abscess','Pustule','Bulla','Patch','Nodule','Ulcer',
          'Crust','Erosion','Excoriation','Atrophy','Exudate','Purpura/Petechiae','Fissure','Induration'
]
concepts_num=len(concepts)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# dataset = ImageFolderWithPaths(
#                     args.image_dir,
#                     transforms.Compose([
#                         transforms.Resize(299),
#                         transforms.CenterCrop(299),
#                         transforms.ToTensor(),
#                         normalize]))
dataset = DDI_Dataset(root=args.image_dir,transform=transforms.Compose([
                         transforms.Resize(299),
                         transforms.CenterCrop(256),
                         transforms.ToTensor(),
                         normalize]),
                         concepts=concepts)
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, shuffle=False,
            num_workers=0, pin_memory=device)
total_size = len(dataset)

train_size = int(0.8 * total_size)
# val_size = int(0.2 * total_size)
val_size = total_size - train_size 
train_set, val_set = random_split(dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=device)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=device)
# test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=device)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
if args.backbone=='RN50':
    feature_size=1000
    backbone = resnet50(pretrained=True)
    # backbone = nn.Sequential(*list(backbone.children())[:-1])
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, feature_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_size, 2),
        nn.Sigmoid()
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
if args.backbone=='vgg':
    feature_size=1000
    backbone = vgg16(pretrained=True)

    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, feature_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_size, 2),
        nn.Sigmoid()
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

if args.backbone=='DenseNet':
    feature_size = 1000
    backbone = densenet169(pretrained=True)
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 128),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(128, concepts_num, args.embedding_size),
    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    # num_features = densenet_model.classifier.in_features
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
if args.backbone=='vit':
    from vit_pytorch import ViT
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
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 128),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(128, concepts_num, args.embedding_size),
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model =Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
if args.backbone=='unet':
    backbone = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    backbone=nn.Sequential(*list(backbone.children())[:-1])
    feature_size=20736
    task_predictor = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, feature_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_size, 2),
        nn.Sigmoid()
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

loss_from_concept_pred=torch.nn.CrossEntropyLoss()
loss_form_c = torch.nn.BCELoss()
loss_form_y = torch.nn.CrossEntropyLoss()
loss_form_neural = torch.nn.BCELoss()
model.train()
best_val_accuracy = 0.0
def lambda_rule(epoch):
    return 0.1 / (1 + epoch)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
for epoch in trange(args.epochs):
    train_y_true = torch.tensor([]).to(device)
    train_y_pred = torch.tensor([]).to(device)
    # train_concepts_true = torch.tensor([]).to(device)
    # train_concepts_pred = torch.tensor([]).to(device)
    all_y_concept_pred=[]
    concept_num_total=[]
     
    for batch in train_dataloader:
        optimizer.zero_grad() 
        path, images, labels, skin_tone,concepts_batch,concept_label=batch
        bs=len(path)
            
        concepts_batch=torch.stack(concepts_batch,dim=1).to(device)
        concepts_batch=concepts_batch.float()
        feature,y_pred,y_pred_neural, c_emb, c_pred,y_pred_concept =model(images.to(device))

        # compute loss

        concept_loss =loss_form_c(c_pred, concepts_batch)
        concept_pred_loss=loss_from_concept_pred(y_pred_concept, F.one_hot(labels.long().ravel()).float().to(device))
        # task_loss = loss_form_y(y_pred.view(-1), labels.float().to(device))
        task_loss = loss_form_y(y_pred, F.one_hot(labels.long().ravel()).float().to(device))
        neural_loss = loss_form_neural(y_pred_neural,F.one_hot(labels.long().ravel()).float().to(device))
        loss = 0.2*concept_loss + 1*task_loss+0.2*neural_loss+0.2*concept_pred_loss
        # train_y_true = torch.cat((train_y_true, labels.view(labels.size(0), -1).to(device)), dim=0)
        train_y_true = torch.cat((train_y_true, F.one_hot(labels.long().ravel()).float().to(device)))
        train_y_pred = torch.cat((train_y_pred, (y_pred.cpu() > 0.5).to(device)), dim=0)
        # train_concepts_true = torch.cat((train_concepts_true, concepts_batch.to(device)), dim=0)
        # train_concepts_pred = torch.cat((train_concepts_pred, (c_pred.cpu() > 0.5).to(device)), dim=0)
        if args.wandb==True:
            wandb.log({'loss':loss,'concept_loss':concept_loss,'concept_pred_loss':concept_pred_loss,'task_loss':task_loss,'neural_loss':neural_loss,'lr':optimizer.param_groups[0]["lr"]})
        
        print(f' loss:{loss} concept_loss:{concept_loss} concept_pred_loss {concept_pred_loss} task_loss {task_loss} neural_loss {neural_loss}')
        loss.backward()
        
        optimizer.step()
    train_task_accuracy = accuracy_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy())
    # train_concept_accuracy = accuracy_score(train_concepts_true.cpu().numpy(), train_concepts_pred.cpu().numpy())
    print( f'train_task_accuracy:{train_task_accuracy}')          
    local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
    global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
    
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
            for batch in tqdm(val_dataloader, desc='val'):
                path, images, labels, skin_tone,concepts_batch,concept_label=batch
                optimizer.zero_grad()      
                concepts_batch=torch.stack(concepts_batch,dim=1).to(device)
                concepts_batch=concepts_batch.float()
                feature,y_pred,y_pred_neural, c_emb, c_pred,c_concept_pred =model(images.to(device))

                local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
                global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
                print(local_explanations,global_explanations)

                # all_y_true.extend(labels.view(labels.size(0), -1).cpu().numpy())
                all_y_true.extend(F.one_hot(labels.long().ravel()).float().cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
                concept_true.append(sum(sum(concepts_batch.cpu().numpy()==(c_pred.cpu().numpy() > 0.5))))
                concept_num_total.append(concepts_batch.shape[0]*concepts_batch.shape[1])
                all_y_concept_pred.extend(c_concept_pred.cpu().numpy() > 0.5)
        model.train()
        # 计算整体的准确度
        task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))
        concept_accuracy = sum(concept_true)/sum(concept_num_total)
        y_concept_pred=accuracy_score(np.array(all_y_true), np.array(all_y_concept_pred))
        
        
        # 计算F1 Score
        task_f1 = f1_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算Precision
        task_precision = precision_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算Recall
        task_recall = recall_score(np.array(all_y_true), np.array(all_y_pred), average='macro')
        # 计算ROC和AUC
        # fpr, tpr, _ = roc_curve(np.array(all_y_true), np.array(all_y_pred))
        task_auc = roc_auc_score(np.array(all_y_true), np.array(all_y_pred))
        if args.wandb==True:
            wandb.log({'task_accuracy':task_accuracy,'concept_accuracy':concept_accuracy,'task_f1':task_f1,'task_precision':task_precision,'task_recall':task_recall,'task_auc':task_auc})
        
        if task_accuracy > best_val_accuracy:
                best_val_accuracy = task_accuracy
                if not os.path.exists(args.savedir):
                    os.mkdir(args.savedir)
                torch.save(model.state_dict(), os.path.join(args.savedir, 'best_model.pth'))
        print('best_val_accuracy:',best_val_accuracy,'task_accuracy:',task_accuracy,'task_f1:',task_f1,'task_precision:',task_precision,'task_recall:',task_recall,'task_auc:',task_auc)
        

best_model_path = os.path.join(args.savedir, 'best_model.pth')
best_model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)
best_model.load_state_dict(torch.load(best_model_path))
from saliency.fullGrad import FullGrad


fullgrad = FullGrad(model)

# Check completeness property
# done automatically while initializing object
fullgrad.checkCompleteness()

# Obtain fullgradient decomposition
input_gradient, bias_gradients = fullgrad.fullGradientDecompose(input_image, target_class)

# Obtain saliency maps
saliency_map = fullgrad.saliency(input_image, target_class)
wandb.finish()

# best_model.eval()
# all_y_true = []
# all_y_pred = []
# all_concepts_true = []
# all_concepts_pred = []
# concept_true=[]
# concept_num_total=[]
# all_y_concept_pred=[]
# with torch.no_grad():
#     for batch in tqdm(test_dataloader, desc='testing!'):
#         path, images, labels, skin_tone,concepts_batch,concept_label=batch
#         optimizer.zero_grad()      
#         concepts_batch=torch.stack(concepts_batch,dim=1).to(device)
#         concepts_batch=concepts_batch.float()
#         feature,y_pred,y_pred_neural, c_emb, c_pred,y_concept_pred =model(images.to(device))

#         local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
#         global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
#         all_y_true.extend(F.one_hot(labels.long().ravel()).float().cpu().numpy())
#         all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
#         concept_true.append(sum(sum(concepts_batch.cpu().numpy()==(c_pred.cpu().numpy() > 0.5))))
#         concept_num_total.append(concepts_batch.shape[0]*concepts_batch.shape[1])
#         all_y_concept_pred.extend(c_concept_pred.cpu().numpy() > 0.5)
        
# task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))
# concept_accuracy = sum(concept_true)/sum(concept_num_total)
# # y_concept_pred=accuracy_score(np.array(all_y_true), np.array(all_y_concept_pred))
# print(f'task_accuracy:{task_accuracy}, concept_accuracy:{concept_accuracy}')
# best_model.to('cpu')
# torch.save(best_model, os.path.join(args.savedir, 'best_model_complete.pth'))
# print('model saved!')


