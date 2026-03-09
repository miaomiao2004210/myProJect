import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model.resnet_versions import ResNet18_v1, ResNet18_v2  # ← 关键！导入你的模型
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# ======================
# 配置路径（请根据实际情况修改）
# ======================
VAL_DIR = "data/plantvillage/val"  # ← 必须和 train_ablation.py 一致
V1_MODEL_PATH = "results/resnet18_v1_scratch_best.pth"
V2_MODEL_PATH = "results/resnet18_v2_scratch_best.pth"
OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 验证集 transform（必须和训练时 val_transform 一致！）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
class_names = val_dataset.classes
num_classes = len(class_names)
print(f"✅ 加载验证集: {num_classes} 个类别")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 评估函数（使用你的自定义模型）
# ======================
def evaluate_model(model_class, model_path):
    model = model_class(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# ======================
# 开始评估
# ======================
print("🔍 评估 ResNet18-v1...")
y_true, y_pred_v1 = evaluate_model(ResNet18_v1, V1_MODEL_PATH)

print("🔍 评估 ResNet18-v2 (+SE)...")
_, y_pred_v2 = evaluate_model(ResNet18_v2, V2_MODEL_PATH)

# ======================
# 计算指标（macro 平均，更公平）
# ======================
def get_metrics(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, prec, rec, f1

acc_v1, prec_v1, rec_v1, f1_v1 = get_metrics(y_true, y_pred_v1)
acc_v2, prec_v2, rec_v2, f1_v2 = get_metrics(y_true, y_pred_v2)

# 打印表格
print("\n" + "="*70)
print("📊 消融实验结果（验证集）")
print("="*70)
print(f"{'Model':<25} {'Acc (%)':<10} {'Prec':<10} {'Recall':<10} {'F1':<10}")
print("-"*70)
print(f"ResNet18-v1               {acc_v1*100:<10.2f} {prec_v1:<10.4f} {rec_v1:<10.4f} {f1_v1:<10.4f}")
print(f"ResNet18-v2 (+SE)         {acc_v2*100:<10.2f} {prec_v2:<10.4f} {rec_v2:<10.4f} {f1_v2:<10.4f}")
print("="*70)

# 保存 JSON
table_data = {
    "ResNet18-v1": {
        "Accuracy": round(acc_v1 * 100, 2),
        "Precision": round(prec_v1, 4),
        "Recall": round(rec_v1, 4),
        "F1": round(f1_v1, 4)
    },
    "ResNet18-v2 (+SE)": {
        "Accuracy": round(acc_v2 * 100, 2),
        "Precision": round(prec_v2, 4),
        "Recall": round(rec_v2, 4),
        "F1": round(f1_v2, 4)
    }
}
with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), 'w', encoding='utf-8') as f:
    json.dump(table_data, f, indent=4, ensure_ascii=False)

# ======================
# 混淆矩阵（v2）
# ======================
print("\n🎨 生成混淆矩阵...")
cm = confusion_matrix(y_true, y_pred_v2)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt='d')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.title("Confusion Matrix (ResNet18-v2)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# ======================
# 各类别 F1 柱状图
# ======================
print("📊 生成各类别 F1 柱状图...")
f1_per_class = f1_score(y_true, y_pred_v2, average=None)

# 简化标签名
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
plt.title("Per-Class F1-Score (ResNet18-v2)", fontsize=14)

# 标出最低的3个
min_idx = np.argsort(f1_per_class)[:3]
for i in min_idx:
    bars[i].set_color('red')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 所有图表已保存至: {os.path.abspath(OUTPUT_DIR)}")