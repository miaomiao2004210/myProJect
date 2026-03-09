# split_dataset.py
import os
import shutil
from sklearn.model_selection import train_test_split

src_dir = "data/plantvillage/color"
dst_dir = "data/plantvillage"

# 创建输出目录
os.makedirs(os.path.join(dst_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(dst_dir, "val"), exist_ok=True)

print("🚀 开始划分数据集...")

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    # 获取图像文件（支持多种格式）
    images = []
    for filename in os.listdir(class_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(filename)

    if len(images) < 5:
        print(f"⚠️ 跳过样本不足的类别：{class_name} ({len(images)} 张)")
        continue

    # 划分训练集和验证集
    train_imgs, val_imgs = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    # 创建子目录
    train_class_dir = os.path.join(dst_dir, "train", class_name)
    val_class_dir = os.path.join(dst_dir, "val", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # 复制训练集
    for img in train_imgs:
        src_img = os.path.join(class_path, img)
        dst_img = os.path.join(train_class_dir, img)
        try:
            shutil.copy(src_img, dst_img)
        except Exception as e:
            print(f"❌ 复制失败：{src_img} -> {dst_img}，错误：{e}")

    # 复制验证集
    for img in val_imgs:
        src_img = os.path.join(class_path, img)
        dst_img = os.path.join(val_class_dir, img)
        try:
            shutil.copy(src_img, dst_img)
        except Exception as e:
            print(f"❌ 复制失败：{src_img} -> {dst_img}，错误：{e}")

print("✅ 数据集划分完成！")