# ======================================
# 精简版：匹配论文改进的轻量化ResNet10 + 单轮训练（100轮）
# 核心改进：ResNet18→ResNet10 + 双重Dropout + 全局平均池化 + 小样本适配
# ======================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision.models.resnet import ResNet, BasicBlock


# -------------------------- 1. 配置参数（适配你的需求：100轮+原train/val划分） --------------------------
class Config:
    IMG_SIZE = 224
    DATASET_ROOT = "E:/Users/hewei/PycharmProjects/pythonProject2/ResNet/da"  # 你的数据集根目录（train/val按类别分文件夹）
    BATCH_SIZE = 16
    LEARNING_RATE = 8e-5
    WEIGHT_DECAY = 2.5e-4
    EPOCHS = 100  # 固定100轮，不再早停
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2
    # 论文双重Dropout参数
    DROPOUT_SPATIAL = 0.25
    DROPOUT_FC = 0.55


# -------------------------- 2. 数据增强（保持和论文一致） --------------------------
target_size = Config.IMG_SIZE if isinstance(Config.IMG_SIZE, tuple) else (Config.IMG_SIZE, Config.IMG_SIZE)
resize_size = (target_size[0] + 32, target_size[1] + 32)

# -------------------------- 修正后：训练集增强（强度减半，避免拟合困难） --------------------------
train_transform = transforms.Compose([
    transforms.Resize(target_size),  # 移除放大+随机裁剪，直接缩放到目标尺寸
    transforms.RandomHorizontalFlip(p=0.3),  # 翻转概率从0.5→0.3
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色扰动从0.2→0.1
    transforms.RandomRotation(5),  # 旋转角度从10°→5°
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.05, scale=(0.02, 0.1))  # 擦除概率从0.1→0.05
])

# -------------------------- 修正后：验证集恢复无增强（保证评估客观，接受小样本偶然1.0） --------------------------
val_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------- 3. 加载原train/val数据集（不再合并拆分） --------------------------
train_dataset = datasets.ImageFolder(
    root=os.path.join(Config.DATASET_ROOT, "train"),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(Config.DATASET_ROOT, "val"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)


# -------------------------- 4. 论文轻量化ResNet10 + 双重Dropout（完全匹配你的结构图/描述） --------------------------
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


def resnet18_simplified(num_classes=Config.NUM_CLASSES, dropout1=Config.DROPOUT_SPATIAL, dropout2=Config.DROPOUT_FC):
    model = LightweightResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=num_classes, dropout_rate=dropout1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout2),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# -------------------------- 5. 训练+验证函数（固定100轮，不再早停） --------------------------
def train_with_monitor():
    # 初始化模型
    model = resnet18_simplified(
        num_classes=Config.NUM_CLASSES,
        dropout1=Config.DROPOUT_SPATIAL,
        dropout2=Config.DROPOUT_FC
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # 保存100轮的所有指标
    train_loss_list = []
    train_acc_list = []
    train_f1_list = []
    val_loss_list = []
    val_acc_list = []
    val_f1_list = []
    val_precision_list = []
    val_recall_list = []

    print(f"开始训练 | 设备：{Config.DEVICE} | 轮数：{Config.EPOCHS}")
    print("=" * 60)

    for epoch in range(1, Config.EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # 训练集指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average="weighted")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # 验证集指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_f1 = f1_score(val_targets, val_preds, average="weighted")
        val_precision = precision_score(val_targets, val_preds, average="weighted", zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average="weighted", zero_division=0)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)

        # 打印每轮结果（清晰展示100轮数据）
        print(f"Epoch {epoch:3d}/{Config.EPOCHS}")
        print(f"Train: Loss={train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}")
        print(
            f"Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f} | Precision={val_precision:.4f} | Recall={val_recall:.4f}")
        print("-" * 50)

        # 每20轮保存一次模型（方便你取10/20/...100轮的结果）
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"resnet10_simplified_epoch_{epoch}.pth")
            print(f"保存模型：resnet10_simplified_epoch_{epoch}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), "results/fast_rcnn_confusion/resnet10_simplified_final_100epoch.pth")
    print("\n训练完成！最终模型已保存")

    # 返回100轮的所有指标，方便你画图/写论文
    return {
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "train_f1": train_f1_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list,
        "val_f1": val_f1_list,
        "val_precision": val_precision_list,
        "val_recall": val_recall_list
    }


# -------------------------- 6. 主入口（一键启动100轮训练） --------------------------
if __name__ == "__main__":
    results = train_with_monitor()
    # 你可以在这里把results保存为npy文件，方便论文画图
    np.save("evaluation_metrics_100epoch.npy", results)
    print("\n100轮训练指标已保存至 evaluation_metrics_100epoch.npy")