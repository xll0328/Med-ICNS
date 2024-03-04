import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
class Neural_Concat_vitbackbone_Model(nn.Module):
    def __init__(self, backbone, concept_encoder, task_predictor_neural, task_predictor):
        super(Neural_Concat_vitbackbone_Model, self).__init__()
        self.backbone = backbone
        self.concept_encoder = concept_encoder
        self.task_predictor_neural = task_predictor_neural
        self.task_predictor = task_predictor

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        x = transforms.ToPILImage()(x)
        feature = self.backbone(x)
        feature = feature['pixel_values'][0]
        # 继续后续的操作
        c_emb, c_pred = self.concept_encoder(feature)
        y_pred_neural = self.task_predictor_neural(c_emb, c_pred)
        y_pred = self.task_predictor(torch.cat((feature, c_emb.view(x.shape[0], -1)), dim=1))
        
        return feature, y_pred, y_pred_neural, c_emb, c_pred
class Neural_Concat_Model(nn.Module):
    def __init__(self, backbone, concept_encoder, task_predictor_neural,task_predictor,task_concept_predictor):
        super(Neural_Concat_Model, self).__init__()
        self.backbone = backbone
        self.concept_encoder = concept_encoder
        self.task_predictor_neural=task_predictor_neural
        self.task_predictor = task_predictor
        self.task_concept_predictor=task_concept_predictor

    def forward(self, x):
        feature = self.backbone(x)
        feature = feature.view(x.size(0), -1)
        c_emb, c_pred = self.concept_encoder(feature)
        y_pred_neural = self.task_predictor_neural(c_emb, c_pred)
        y_pred_concept = self.task_concept_predictor(c_emb.reshape(len(c_emb), -1))
        y_pred=self.task_predictor(torch.cat((feature,c_emb.view(x.shape[0],-1)),dim=1))
        return feature,y_pred,y_pred_neural, c_emb, c_pred,y_pred_concept

class ResNetModel(nn.Module):
    def __init__(self, resnet_model,task_predictor):
        super(ResNetModel, self).__init__()
        self.resnet_model = resnet_model
        self.task_predictor = task_predictor

    def forward(self, x):
        feature = self.resnet_model(x)
        feature = feature.view(feature.size(0), -1)
        y_pred=self.task_predictor(feature)
        return y_pred

class DenseNetModel(nn.Module):
    def __init__(self, densenet_model,task_predictor):
        super(DenseNetModel, self).__init__()
        self.densenet_model = densenet_model
        self.task_predictor = task_predictor

    def forward(self, x):
        feature = self.densenet_model(x)
        feature = feature.view(feature.size(0), -1)
        y_pred=self.task_predictor(feature)
        return y_pred
    
class VitModel(nn.Module):
    def __init__(self, vid_model,task_predictor):
        super(DenseNetModel, self).__init__()
        self.vid_model = vid_model
        self.task_predictor = task_predictor

    def forward(self, x):
        feature = self.vid_model(x)
        feature = feature.view(feature.size(0), -1)
        y_pred=self.task_predictor(feature)
        return y_pred

