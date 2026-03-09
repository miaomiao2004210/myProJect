import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.resnet_versions import ResNet18_v1, ResNet18_v2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import csv

# ======================
# 配置
# ======================
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38

# ✅ 本地定义 pin_memory，不再从 train.py 导入
pin_memory = DEVICE.type == "cuda"

os.makedirs("results", exist_ok=True)

# 数据增强
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/plantvillage/train", transform=transform)
val_dataset = datasets.ImageFolder("data/plantvillage/val", transform=val_transform)
class_names = val_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_memory)


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def train_model(model, model_name):
    print(f"\n🚀 开始从头训练 {model_name} (Scratch Training)...")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        correct_train, total_train = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{EPOCHS}", leave=False):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            correct_train += (pred == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        acc_train = 100 * correct_train / total_train

        # 验证
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                correct_val += (pred == labels).sum().item()
                total_val += labels.size(0)
        acc_val = 100 * correct_val / total_val

        history.append({
            'epoch': epoch + 1,
            'train_acc': acc_train,
            'val_acc': acc_val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), f"results/{model_name}_best.pth")

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{model_name}] Epoch {epoch + 1}: "
                  f"Train Acc={acc_train:.2f}%, Val Acc={acc_val:.2f}%, LR={current_lr:.5f}")

    with open(f"results/{model_name}_history.json", 'w') as f:
        json.dump(history, f)

    print(f"✅ {model_name} 从头训练完成！最佳验证准确率: {best_acc:.2f}%")
    return best_acc, history


def plot_training_curves(history_v1, history_v2):
    epochs = [h['epoch'] for h in history_v1]
    train_acc_v1 = [h['train_acc'] for h in history_v1]
    val_acc_v1 = [h['val_acc'] for h in history_v1]
    train_acc_v2 = [h['train_acc'] for h in history_v2]
    val_acc_v2 = [h['val_acc'] for h in history_v2]
    lr_v1 = [h['lr'] for h in history_v1]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_v1, 'b-', label='v1 Train Acc')
    plt.plot(epochs, val_acc_v1, 'b--', label='v1 Val Acc')
    plt.plot(epochs, train_acc_v2, 'g-', label='v2 (+SE) Train Acc')
    plt.plot(epochs, val_acc_v2, 'g--', label='v2 (+SE) Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, lr_v1, 'r-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/training_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_bar_chart(acc_v1, prec_v1, rec_v1, f1_v1,
                           acc_v2, prec_v2, rec_v2, f1_v2):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    v1_scores = [acc_v1, prec_v1, rec_v1, f1_v1]
    v2_scores = [acc_v2, prec_v2, rec_v2, f1_v2]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, v1_scores, width, label='ResNet18-v1.0', color='#1f77b4')
    plt.bar(x + width/2, v2_scores, width, label='ResNet18-v2.0 (+SE)', color='#ff7f0e')

    plt.ylabel('Score')
    plt.title('Model Performance Comparison (Weighted)')
    plt.xticks(x, metrics)
    plt.ylim(0.85, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标签
    for i in range(len(metrics)):
        plt.text(i - width/2, v1_scores[i] + 0.002, f'{v1_scores[i]:.4f}', ha='center')
        plt.text(i + width/2, v2_scores[i] + 0.002, f'{v2_scores[i]:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(labels, preds_v1, preds_v2):
    cm_v1 = confusion_matrix(labels, preds_v1)
    cm_v2 = confusion_matrix(labels, preds_v2)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(cm_v1, ax=axes[0], cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    axes[0].set_title("Confusion Matrix: ResNet18-v1.0", fontsize=14)

    sns.heatmap(cm_v2, ax=axes[1], cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title("Confusion Matrix: ResNet18-v2.0 (+SE)", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(labels, preds_v1, preds_v2):
    f1_v1 = f1_score(labels, preds_v1, average=None)
    f1_v2 = f1_score(labels, preds_v2, average=None)

    short_names = []
    for name in class_names:
        plant, disease = name.split('___')
        disease = disease.replace('_', ' ').replace('(', '').replace(')', '')
        short_names.append(f"{plant[:4]}\n{disease[:12]}")

    x = np.arange(len(f1_v1))
    width = 0.35

    plt.figure(figsize=(20, 6))
    plt.bar(x - width/2, f1_v1, width, label='v1.0', color='#1f77b4')
    plt.bar(x + width/2, f1_v2, width, label='v2.0 (+SE)', color='#ff7f0e')

    plt.xticks(x, short_names, rotation=90, fontsize=8)
    plt.ylabel('F1-Score')
    plt.title('Per-Class F1-Score Comparison')
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("results/per_class_f1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Train v1
    model_v1 = ResNet18_v1(num_classes=NUM_CLASSES)
    acc_v1, history_v1 = train_model(model_v1, "resnet18_v1_scratch")
    model_v1.load_state_dict(torch.load("results/resnet18_v1_scratch_best.pth"))
    labels, preds_v1 = evaluate_model(model_v1, val_loader)
    prec_v1 = precision_score(labels, preds_v1, average='weighted')
    rec_v1 = recall_score(labels, preds_v1, average='weighted')
    f1_v1 = f1_score(labels, preds_v1, average='weighted')

    # Train v2
    model_v2 = ResNet18_v2(num_classes=NUM_CLASSES)
    acc_v2, history_v2 = train_model(model_v2, "resnet18_v2_scratch")
    model_v2.load_state_dict(torch.load("results/resnet18_v2_scratch_best.pth"))
    labels, preds_v2 = evaluate_model(model_v2, val_loader)
    prec_v2 = precision_score(labels, preds_v2, average='weighted')
    rec_v2 = recall_score(labels, preds_v2, average='weighted')
    f1_v2 = f1_score(labels, preds_v2, average='weighted')

    # 打印结果
    print("\n" + "=" * 80)
    print("📊 消融实验结果（从头训练）:")
    print(f"{'Model':<25} {'Acc (%)':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    print(f"ResNet18-v1.0            {acc_v1:<10.2f} {f1_v1:<10.4f} {prec_v1:<10.4f} {rec_v1:<10.4f}")
    print(f"ResNet18-v2.0 (+SE)      {acc_v2:<10.2f} {f1_v2:<10.4f} {prec_v2:<10.4f} {rec_v2:<10.4f}")
    print("=" * 80)

    # 保存 CSV
    with open("results/ablation.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Accuracy (%)", "F1-Score", "Precision", "Recall"])
        writer.writerow(["ResNet18-v1.0", round(acc_v1, 2), round(f1_v1, 4), round(prec_v1, 4), round(rec_v1, 4)])
        writer.writerow(["ResNet18-v2.0 (+SE)", round(acc_v2, 2), round(f1_v2, 4), round(prec_v2, 4), round(rec_v2, 4)])

    # 生成所有图表
    plot_training_curves(history_v1, history_v2)
    plot_metrics_bar_chart(acc_v1/100, prec_v1, rec_v1, f1_v1, acc_v2/100, prec_v2, rec_v2, f1_v2)
    plot_confusion_matrices(labels, preds_v1, preds_v2)
    plot_per_class_f1(labels, preds_v1, preds_v2)

    print("\n✅ 所有图表已保存至 'results/' 目录！")
    print("📁 包含：")
    print("   - training_curves_comparison.png")
    print("   - metrics_comparison.png          ← 【包含 Acc, Prec, Rec, F1 四项对比】")
    print("   - confusion_matrices_comparison.png")
    print("   - per_class_f1_comparison.png")
    print("   - ablation.csv")


if __name__ == '__main__':
    main()