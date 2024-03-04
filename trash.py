from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

model_name = "google/vit-base-patch16-224-in21k"

# 创建 ViT 模型和特征提取器
feature_extractor = ViTFeatureExtractor(model_name)
vit_model = ViTForImageClassification.from_pretrained(model_name)

# 去掉分类头部分，获取 backbone
vit_backbone = torch.nn.Sequential(*(list(vit_model.children())[:-1]))

# 定义输入图像
input_image = torch.randn(3, 224, 224)  # 3 通道的 224x224 输入，可以根据你的需要更改

# 使用 torchvision 进行图像预处理和归一化
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

input_image = preprocess(input_image)

# 将图像像素值归一化到 [0, 1]
input_image = input_image / 255.0

# 使用特征提取器处理输入
inputs = feature_extractor(images=input_image, return_tensors="pt")

# 前向传播获取输出
output_feature = vit_backbone(inputs.pixel_values).last_hidden_state

# 打印输出的形状
print("Backbone Output Shape:", output_feature.shape)
vit_model = ViTForImageClassification.from_pretrained(model_name)

# 打印模型的结构，查看 MLP 的位置
print(vit_model)

# 如果你知道 MLP 层的名称，你可以直接通过名称获取
mlp_layer = vit_model.mlp  # 这里的 "mlp" 是示例，请替换为实际的层名称

# 打印 MLP 层的结构
print(mlp_layer)