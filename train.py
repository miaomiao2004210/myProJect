import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.resnet_versions import ResNet18SE
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import json

# ======================
# 配置
# ======================
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38

# 创建输出目录
os.makedirs("model", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ======================
# 数据预处理
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# 数据加载（GPU 优化版）
# ======================
train_dataset = datasets.ImageFolder("data/plantvillage/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/plantvillage/val", transform=val_transform)

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=pin_memory
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=pin_memory
)

class_names = val_dataset.classes
print(f"✅ 检测到 {len(class_names)} 个类别")

# ======================
# 模型、损失、优化器、调度器
# ======================
model = ResNet18SE(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 🔧 修复：移除 verbose=True（兼容旧版 PyTorch）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
    # verbose=True  # ← 不支持，已移除
)

# ======================
# 训练主函数
# ======================
def main():
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"🚀 使用设备: {DEVICE}")
    print(f"📊 训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # ---------- 训练 ----------
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        acc_train = 100 * correct_train / total_train

        # ---------- 验证 ----------
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        acc_val = 100 * correct_val / total_val

        # ---------- 调度 & 手动打印 LR 变化 ----------
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(acc_val)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"📉 学习率调整: {old_lr:.2e} → {new_lr:.2e}")

        # ---------- 保存最佳模型 ----------
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), "model/resnet18_se_best.pth")
            print(f"🎉 保存新最佳模型！验证准确率: {acc_val:.2f}%")

        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(acc_train)
        val_accuracies.append(acc_val)

        print(f"[{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {acc_train:.2f}% | "
              f"Val Acc: {acc_val:.2f}%")

    # ======================
    # 训练完成，开始生成图表
    # ======================
    print("\n🎨 正在生成毕设所需图表...")

    # 1. 训练曲线
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'g-o', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-o', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 加载最佳模型进行最终评估
    model.load_state_dict(torch.load("model/resnet18_se_best.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 3. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt='d')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 各类别 F1 柱状图
    f1_per_class = f1_score(y_true, y_pred, average=None)
    short_names = []
    for name in class_names:
        plant, disease = name.split('___')
        disease = disease.replace('_', ' ').replace('(', '').replace(')', '')
        short_names.append(f"{plant[:4]}\n{disease[:12]}")

    plt.figure(figsize=(18, 6))
    bars = plt.bar(range(len(f1_per_class)), f1_per_class, color='#4CAF50')
    plt.xticks(range(len(f1_per_class)), short_names, rotation=90, fontsize=8)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0.85, 1.0)
    plt.title("Per-Class F1-Score", fontsize=14)

    # 标出最低的3个类别（红色）
    min_idx = np.argsort(f1_per_class)[:3]
    for i in min_idx:
        bars[i].set_color('red')

    plt.tight_layout()
    plt.savefig("figures/per_class_f1.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 打印最终指标（使用 macro 平均）
    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n" + "="*70)
    print("📊 最终模型性能（验证集，macro 平均）")
    print("="*70)
    print(f"{'Accuracy':<12}: {acc*100:.2f}%")
    print(f"{'Precision':<12}: {prec:.4f}")
    print(f"{'Recall':<12}: {rec:.4f}")
    print(f"{'F1-Score':<12}: {f1:.4f}")
    print("="*70)

    # 6. 保存指标到 JSON
    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }
    with open("figures/final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n✅ 所有图表和指标已保存至 'figures/' 目录！")
    print("📁 包含：")
    print("   - training_curves.png")
    print("   - confusion_matrix.png")
    print("   - per_class_f1.png")
    print("   - final_metrics.json")
    print("\n💡 提示：所有图像均为 300 DPI，可直接用于毕业论文插图！")


if __name__ == "__main__":
    main()