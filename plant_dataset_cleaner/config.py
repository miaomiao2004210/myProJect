# config.py
import re

# ✅ 完整 PlantVillage 类别映射（38类）
CATEGORY_PREFIX_MAP = {
    "Apple___Apple_scab": "apple_apple_scab",
    "Apple___Black_rot": "apple_black_rot",
    "Apple___Cedar_apple_rust": "apple_cedar_apple_rust",
    "Apple___healthy": "apple_healthy",
    "Blueberry___healthy": "blueberry_healthy",
    "Cherry_(including_sour)___healthy": "cherry_healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry_powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn_gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "corn_common_rust",
    "Corn_(maize)___healthy": "corn_healthy",
    "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_leaf_blight",
    "Grape___Black_rot": "grape_black_rot",
    "Grape___Esca_(Black_Measles)": "grape_esca",
    "Grape___healthy": "grape_healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
    "Orange___Haunglongbing_(Citrus_greening)": "orange_huanglongbing",
    "Peach___Bacterial_spot": "peach_bacterial_spot",
    "Peach___healthy": "peach_healthy",
    "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper,_bell___healthy": "pepper_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___healthy": "potato_healthy",
    "Potato___Late_blight": "potato_late_blight",
    "Raspberry___healthy": "raspberry_healthy",
    "Soybean___healthy": "soybean_healthy",
    "Squash___Powdery_mildew": "squash_powdery_mildew",
    "Strawberry___healthy": "strawberry_healthy",
    "Strawberry___Leaf_scorch": "strawberry_leaf_scorch",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___healthy": "tomato_healthy",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato_tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus"
}

# ✅ 精准噪声模式（覆盖 JR_Frg, UF.GRC, FREC 等）
NOISE_PATTERNS = [
    # UUID___ 前缀
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}___',

    # 实验代号（精准匹配）
    r'JR_Frg\.E\.S\s*\d+',
    r'UF\.GRC_BS_Lab\s+Leaf\s+\d+',
    r'FREC_Scab_\d+',
    r'PS\d+_\d+',
    r'R\d+_',
    r'GHLB\s+Leaf\s+\d+(\.\d+)?\s+Day\s+\d+',

    # 通用模式
    r'\bLeaf\s*\d+(\.\d+)?\b',
    r'\bDay\s+\d+\b',

    # 移除多余符号
    r'[^\w\s\-]',  # 只保留字母、数字、空格、连字符
]


def clean_filename(filename: str, category_key: str) -> str:
    """安全清洗文件名"""
    name = filename
    ext = ""
    if '.' in name:
        name, ext = name.rsplit('.', 1)
        ext = '.' + ext.lower()

    # 应用噪声规则（替换为空格，避免下划线爆炸）
    for pattern in NOISE_PATTERNS:
        name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)

    # 清理多余空格和非法字符
    name = re.sub(r'\s+', ' ', name)  # 多空格 → 单空格
    name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name)  # 只保留合法字符
    name = re.sub(r'\s+', '_', name).strip('_')  # 空格 → 下划线

    # 获取前缀
    prefix = CATEGORY_PREFIX_MAP.get(category_key, "unknown")
    new_name = f"{prefix}_{name}{ext}" if name else f"{prefix}{ext}"

    # 🔒 Windows 安全长度（最大 255 字符，留出路径空间）
    if len(new_name) > 180:
        stem = new_name[:150]
        suffix = new_name.split('.')[-1]
        new_name = f"{stem}_shortened.{suffix}"

    return new_name