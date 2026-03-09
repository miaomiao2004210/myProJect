# check_missing_categories.py
from pathlib import Path
from config import CATEGORY_PREFIX_MAP

dataset_dir = "D:/pythonProject/data/plantvillage/color"  # 修改为你的路径

actual_categories = {d.name for d in Path(dataset_dir).iterdir() if d.is_dir()}
defined_categories = set(CATEGORY_PREFIX_MAP.keys())

missing = actual_categories - defined_categories
extra = defined_categories - actual_categories

if missing:
    print("❌ 以下类别未在 config.py 中定义:")
    for cat in sorted(missing):
        print(f"  - '{cat}'")
else:
    print("✅ 所有类别均已定义！")

if extra:
    print(f"\nℹ️  config.py 中有 {len(extra)} 个未使用的类别（可忽略）")