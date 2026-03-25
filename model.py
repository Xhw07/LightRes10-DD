import torch
import torch.nn as nn
from torchvision import models


# 最简单的ResNet创建方式
def simple_resnet():
    # 尝试不同的加载方式
    try:
        model = models.resnet50(pretrained=False)  # 不使用预训练
        print("创建随机初始化的ResNet-50")
    except Exception as e:
        print(f"错误: {e}")
        return None

    # 修改最后一层
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# 测试
if __name__ == "__main__":
    model = simple_resnet()
    if model:
        print("模型创建成功!")
        print(f"输出层维度: {model.fc.out_features}")
    else:
        print("模型创建失败!")
