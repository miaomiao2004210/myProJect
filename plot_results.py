# plot_results.py
import json
import matplotlib.pyplot as plt

# 读取训练历史
with open('results/resnet18_v1_history.json') as f:
    hist1 = json.load(f)
with open('results/resnet18_v2_history.json') as f:
    hist2 = json.load(f)

# 提取数据
epochs = [h['epoch'] for h in hist1]
val1 = [h['val_acc'] for h in hist1]
val2 = [h['val_acc'] for h in hist2]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, val1, label='ResNet18-v1.0 (Baseline)', linewidth=2, marker='o', markersize=3)
plt.plot(epochs, val2, label='ResNet18-v2.0 (+SE)', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title('Ablation Study: Effect of SE Module', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存并显示
plt.savefig('results/ablation_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('results/ablation_comparison.pdf')  # 可选：用于论文的矢量图
plt.show()