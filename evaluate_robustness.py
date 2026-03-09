# evaluate_robustness.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, f1_score
from model.resnet_versions import ResNet18_v1, ResNet18_v2
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38

# 数据预处理（与训练一致）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def evaluate_model(model, model_path, test_loader, dataset_name):
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n📊 {dataset_name} 评估结果:")
    print(f"  准确率: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  宏平均 F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"  错误率: {(1 - acc):.4f} ({(1 - acc) * 100:.2f}%)")

    return acc, macro_f1


if __name__ == '__main__':
    # 加载测试集
    standard_val = DataLoader(
        datasets.ImageFolder("data/plantvillage/val", transform=val_transform),
        batch_size=32, shuffle=False
    )

    complex_test = DataLoader(
        datasets.ImageFolder("data/plantvillage/test_complex", transform=val_transform),
        batch_size=32, shuffle=False
    )

    # 评估 v1.0
    model_v1 = ResNet18_v1(num_classes=NUM_CLASSES)
    acc1_std, f11_std = evaluate_model(model_v1, "results/resnet18_v1_scratch_best.pth", standard_val,
                                       "标准验证集 (v1.0)")
    acc1_cmp, f11_cmp = evaluate_model(model_v1, "results/resnet18_v1_scratch_best.pth", complex_test,
                                       "复杂测试集 (v1.0)")

    # 评估 v2.0
    model_v2 = ResNet18_v2(num_classes=NUM_CLASSES)
    acc2_std, f12_std = evaluate_model(model_v2, "results/resnet18_v2_scratch_best.pth", standard_val,
                                       "标准验证集 (v2.0)")
    acc2_cmp, f12_cmp = evaluate_model(model_v2, "results/resnet18_v2_scratch_best.pth", complex_test,
                                       "复杂测试集 (v2.0)")

    # 打印对比
    print("\n" + "=" * 70)
    print("🔍 鲁棒性对比（错误率）:")
    print(f"ResNet18-v1.0: 标准={1 - acc1_std:.4f}, 复杂={1 - acc1_cmp:.4f}")
    print(f"ResNet18-v2.0: 标准={1 - acc2_std:.4f}, 复杂={1 - acc2_cmp:.4f}")
    print(
        f"→ 在复杂场景下，v2.0 错误率降低: {(1 - acc1_cmp) - (1 - acc2_cmp):.4f} ({((1 - acc1_cmp) - (1 - acc2_cmp)) / (1 - acc1_cmp) * 100:.1f}%)")
    print("=" * 70)