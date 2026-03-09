# analyze_results.py
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.resnet_versions import ResNet18_v1, ResNet18_v2

# ======================
# 配置（与训练一致）
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
BATCH_SIZE = 32

# 数据路径（只加载验证集）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder("data/plantvillage/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
class_names = val_dataset.classes


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


def load_history(model_name):
    with open(f"results/{model_name}_history.json", 'r') as f:
        return json.load(f)


def plot_training_curves(history_v1, history_v2):
    # （复用你原来的函数）
    epochs = [h['epoch'] for h in history_v1]
    train_acc_v1 = [h['train_acc'] for h in history_v1]
    val_acc_v1 = [h['val_acc'] for h in history_v1]
    train_acc_v2 = [h['train_acc'] for h in history_v2]
    val_acc_v2 = [h['val_acc'] for h in history_v2]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_acc_v1, 'b--', label='v1 Val Acc')
    plt.plot(epochs, val_acc_v2, 'g--', label='v2 (+SE) Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True)
    plt.savefig("results/post_analysis_val_acc.png", dpi=300)
    plt.close()


def main():
    print("🔍 开始训练后分析...")

    # 加载历史记录
    history_v1 = load_history("resnet18_v1_scratch")
    history_v2 = load_history("resnet18_v2_scratch")

    # 加载模型并评估
    model_v1 = ResNet18_v1(num_classes=NUM_CLASSES).to(DEVICE)
    model_v1.load_state_dict(torch.load("results/resnet18_v1_scratch_best.pth", map_location=DEVICE))
    labels, preds_v1 = evaluate_model(model_v1, val_loader)

    model_v2 = ResNet18_v2(num_classes=NUM_CLASSES).to(DEVICE)
    model_v2.load_state_dict(torch.load("results/resnet18_v2_scratch_best.pth", map_location=DEVICE))
    _, preds_v2 = evaluate_model(model_v2, val_loader)

    # 计算指标
    acc_v1 = 100 * sum(np.array(labels) == np.array(preds_v1)) / len(labels)
    acc_v2 = 100 * sum(np.array(labels) == np.array(preds_v2)) / len(labels)

    prec_v1 = precision_score(labels, preds_v1, average='weighted')
    rec_v1 = recall_score(labels, preds_v1, average='weighted')
    f1_v1 = f1_score(labels, preds_v1, average='weighted')

    prec_v2 = precision_score(labels, preds_v2, average='weighted')
    rec_v2 = recall_score(labels, preds_v2, average='weighted')
    f1_v2 = f1_score(labels, preds_v2, average='weighted')

    # 打印结果
    print("\n📊 训练后分析结果:")
    print(f"ResNet18-v1.0      → Acc: {acc_v1:.2f}%, F1: {f1_v1:.4f}")
    print(f"ResNet18-v2.0 (+SE)→ Acc: {acc_v2:.2f}%, F1: {f1_v2:.4f}")

    # 画图
    plot_training_curves(history_v1, history_v2)
    print("✅ 分析完成！图表已保存。")


if __name__ == '__main__':
    main()