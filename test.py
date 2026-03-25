# ======================================
# 改进ResNet10 测试代码（分类别+整体指标）
# 适配你的训练日志/模型结构，直接输出雷达图所需数据
# ======================================
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)
from torchvision.models.resnet import ResNet, BasicBlock


# -------------------------- 1. 核心配置（直接匹配你的训练环境） --------------------------
class TestConfig:
    IMG_SIZE = 224  # 和训练时一致
    DATASET_ROOT = "E:/Users/hewei/PycharmProjects/pythonProject2/ResNet/da/test"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2  # 二分类（毕业预警/正常毕业）
    # 模型权重路径（训练日志中最终保存的模型）
    MODEL_WEIGHT_PATH = "/ResNet/results/fast_rcnn_confusion/resnet10_simplified_final_100epoch.pth"
    # 匹配训练时的Dropout参数
    DROPOUT_SPATIAL = 0.2
    DROPOUT_FC = 0.5


# -------------------------- 2. 数据预处理（和训练时val/test完全一致） --------------------------
target_size = TestConfig.IMG_SIZE if isinstance(TestConfig.IMG_SIZE, tuple) else (
TestConfig.IMG_SIZE, TestConfig.IMG_SIZE)
test_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------- 3. 加载test数据集（静态划分后的test集） --------------------------
test_dataset = datasets.ImageFolder(
    root="E:/Users/hewei/PycharmProjects/pythonProject2/ResNet/da/test",  # 就是这里写错了
    transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,  # 适配小样本，避免批次波动
    shuffle=False,
    num_workers=0
)

# 打印test集基础信息（确认类别顺序）
print("=" * 80)
print("📋 测试集配置信息")
print(f"测试集样本总数：{len(test_dataset)}")
print(f"类别顺序（对应后续指标）：{test_dataset.classes}")  # 格式：[毕业预警类, 正常毕业类]
print(f"使用设备：{TestConfig.DEVICE}")
print(f"待加载模型：{TestConfig.MODEL_WEIGHT_PATH}")
print("=" * 80)


# -------------------------- 4. 模型定义（和训练时完全一致，确保权重加载成功） --------------------------
class LightweightResNet(ResNet):
    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1], num_classes=2, dropout_rate=0.2):
        super().__init__(block, layers, num_classes=num_classes, norm_layer=nn.BatchNorm2d)
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, layers[0], dropout_rate=dropout_rate, spatial_dropout=True)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dropout_rate=dropout_rate, spatial_dropout=True)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dropout_rate=dropout_rate, spatial_dropout=True)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dropout_rate=dropout_rate, spatial_dropout=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dropout_rate=0.2, spatial_dropout=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        if spatial_dropout:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
            if spatial_dropout:
                layers.append(nn.Dropout2d(p=dropout_rate))
        return nn.Sequential(*layers)


def resnet18_simplified(num_classes=TestConfig.NUM_CLASSES, dropout1=TestConfig.DROPOUT_SPATIAL,
                        dropout2=TestConfig.DROPOUT_FC):
    model = LightweightResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=num_classes, dropout_rate=dropout1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout2),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# -------------------------- 5. 核心：分类别+整体指标计算函数 --------------------------
def calculate_metrics(targets, preds, classes):
    """
    输入：真实标签、预测标签、类别名称列表
    输出：分类别指标字典 + 整体指标字典
    """
    # 1. 分类别指标（每个类别单独计算）
    cls_precision = precision_score(targets, preds, average=None, zero_division=0)
    cls_recall = recall_score(targets, preds, average=None, zero_division=0)
    cls_f1 = f1_score(targets, preds, average=None, zero_division=0)

    # 2. 混淆矩阵 → 计算分类别特异性（Specificity = TN/(TN+FP)）
    cm = confusion_matrix(targets, preds)
    # 处理极端情况（单类别样本导致混淆矩阵维度异常）
    if cm.shape != (2, 2):
        cm = np.array([[len(targets) - sum(preds), 0], [0, sum(preds)]])
    TN, FP, FN, TP = cm.ravel()  # 二分类混淆矩阵固定展开顺序：TN(0,0), FP(0,1), FN(1,0), TP(1,1)

    # 分类别特异性（对应两个类别）
    spec_cls = [
        TN / (TN + FP) if (TN + FP) != 0 else 0.0,  # 类别0（毕业预警）的特异性
        TP / (TP + FN) if (TP + FN) != 0 else 0.0  # 类别1（正常毕业）的特异性
    ]

    # 3. 整体加权指标（和训练日志一致）
    overall_acc = np.mean(targets == preds)
    overall_precision = precision_score(targets, preds, average="weighted", zero_division=0)
    overall_recall = recall_score(targets, preds, average="weighted", zero_division=0)
    overall_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    overall_spec = (TN + TP) / (TN + FP + FN + TP) if (TN + FP + FN + TP) != 0 else 0.0  # 整体特异性

    # 封装分类别指标（绑定类别名称，方便识别）
    cls_metrics = {
        classes[0]: {  # 毕业预警类
            "Precision": round(cls_precision[0], 4),
            "Recall": round(cls_recall[0], 4),
            "Specificity": round(spec_cls[0], 4),
            "F1-score": round(cls_f1[0], 4)
        },
        classes[1]: {  # 正常毕业类
            "Precision": round(cls_precision[1], 4),
            "Recall": round(cls_recall[1], 4),
            "Specificity": round(spec_cls[1], 4),
            "F1-score": round(cls_f1[1], 4)
        }
    }

    # 封装整体指标
    overall_metrics = {
        "Accuracy": round(overall_acc, 4),
        "Precision": round(overall_precision, 4),
        "Recall": round(overall_recall, 4),
        "Specificity": round(overall_spec, 4),
        "F1-score": round(overall_f1, 4),
        "ConfusionMatrix": cm.tolist()
    }

    return cls_metrics, overall_metrics


# -------------------------- 6. 加载模型 + 执行测试 --------------------------
# 初始化模型
model = resnet18_simplified(
    num_classes=TestConfig.NUM_CLASSES,
    dropout1=TestConfig.DROPOUT_SPATIAL,
    dropout2=TestConfig.DROPOUT_FC
).to(TestConfig.DEVICE)

# 加载训练好的权重
try:
    model.load_state_dict(torch.load(TestConfig.MODEL_WEIGHT_PATH, map_location=TestConfig.DEVICE))
    print(f"✅ 成功加载模型权重：{TestConfig.MODEL_WEIGHT_PATH}")
except Exception as e:
    print(f"❌ 权重加载失败！错误信息：{e}")
    print("⚠️  请检查：1. 模型路径是否正确；2. 模型结构是否和训练时一致")
    exit()

# 测试推理（无梯度计算）
model.eval()
test_preds = []
test_targets = []

print("\n🚀 开始测试推理...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(TestConfig.DEVICE), labels.to(TestConfig.DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)  # 取预测概率最大的类别
        # 累计预测结果和真实标签
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# 转换为numpy数组，方便计算指标
test_preds = np.array(test_preds)
test_targets = np.array(test_targets)

# -------------------------- 7. 计算并输出指标（论文可用） --------------------------
cls_metrics, overall_metrics = calculate_metrics(
    test_targets, test_preds, test_dataset.classes
)

print("\n" + "=" * 80)
print("📊 最终测试集指标（分类别+整体，可直接复制到Origin）")
print("=" * 80)

# 打印分类别指标
print("\n🎯 分类别指标：")
for cls_name, metrics in cls_metrics.items():
    print(f"\n【{cls_name}】")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

# 打印整体指标
print("\n\n✅ 整体加权指标：")
for metric_name, value in overall_metrics.items():
    if metric_name != "ConfusionMatrix":
        print(f"  {metric_name}: {value}")

# 打印混淆矩阵
print(f"\n🔍 混淆矩阵（TN, FP, FN, TP）：")
print(f"  {overall_metrics['ConfusionMatrix']}")
print("=" * 80)

# -------------------------- 8. 保存指标到文件（后续可复用） --------------------------
save_data = {
    "类别顺序": test_dataset.classes,
    "分类别指标": cls_metrics,
    "整体指标": overall_metrics,
    "预测标签": test_preds.tolist(),
    "真实标签": test_targets.tolist()
}
np.save("test_metrics_with_classes.npy", save_data)
print(f"\n💾 指标已保存到文件：test_metrics_with_classes.npy")
print("=" * 80)