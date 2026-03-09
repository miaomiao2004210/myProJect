# clean_dataset_cli.py
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from config import clean_filename, CATEGORY_PREFIX_MAP


def backup_dataset(root_dir: str):
    """安全备份"""
    backup_parent = Path(root_dir).parent / "backup"
    backup_parent.mkdir(exist_ok=True)
    backup_dir = backup_parent / Path(root_dir).name

    if not backup_dir.exists():
        print(f"正在备份数据集到 {backup_dir} ...")
        shutil.copytree(root_dir, backup_dir)
        print("✅ 备份完成！原始数据已安全保存。")
    else:
        print(f"⚠️ 备份已存在，跳过备份。")


def clean_dataset(root_dir: str, dry_run: bool = False):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"❌ 数据集目录不存在: {root_dir}")

    if not dry_run:
        backup_dataset(root_dir)

    total_renamed = 0
    errors = []
    unknown_categories = set()

    for category_dir in root_path.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        if category_name not in CATEGORY_PREFIX_MAP:
            unknown_categories.add(category_name)
            continue

        print(f"\n🔄 处理类别: {category_name}")
        files = [f for f in category_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        for img_file in tqdm(files, desc="重命名"):
            try:
                old_name = img_file.name
                new_name = clean_filename(old_name, category_name)
                new_path = img_file.parent / new_name

                # 避免重名
                counter = 1
                base_new_path = new_path
                while new_path.exists():
                    stem = base_new_path.stem
                    suffix = base_new_path.suffix
                    new_path = img_file.parent / f"{stem}_{counter}{suffix}"
                    counter += 1

                if not dry_run:
                    img_file.rename(new_path)
                total_renamed += 1

            except Exception as e:
                errors.append((str(img_file), str(e)))

    # 报告未知类别
    if unknown_categories:
        print(f"\n⚠️ 跳过了 {len(unknown_categories)} 个未知类别:")
        for cat in sorted(unknown_categories):
            print(f"  - {cat}")
        print("\n💡 请在 config.py 的 CATEGORY_PREFIX_MAP 中添加这些类别！")

    print(f"\n✅ 完成！共重命名 {total_renamed} 个文件。")
    if errors:
        print(f"❌ 出现 {len(errors)} 个错误，请检查:")
        for err in errors[:5]:
            print(f"  {err[0]}: {err[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="植物病害数据集清洗工具（修复版）")
    parser.add_argument("dataset_dir", help="数据集根目录路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际重命名")

    args = parser.parse_args()
    clean_dataset(args.dataset_dir, dry_run=args.dry_run)