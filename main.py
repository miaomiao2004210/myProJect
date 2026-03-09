# main.py,基于 Flask 的 Web 服务程序,完整的植物病害识别 Web 应用
import os
import torch
from torchvision import transforms, models
from PIL import Image
import sqlite3
import json
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# 配置路径
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
DATABASE = 'database.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            predicted_class TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 加载模型
model = models.mobilenet_v2(pretrained=False)
num_classes = 38  # 修改为你实际的类别数
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('model/model.pth'))  # 替换为你的模型路径
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 预测函数
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # 添加 batch 维度
    input_tensor = input_tensor.to('cpu')

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted_idx.item()].item()

    class_names = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___Columnar", "Apple___Gray_mold", "Apple___Leaf_blight",
        "Apple___Powdery_mildew", "Apple___Scab", "Apple___Septoria_leaf_spot",
        "Apple___Stem_end_rot", "Apple___White_spot", "Apple___Xanthomonas",
        "Cherry___Peach_blossom_blight", "Cherry___Peach_brown_spot",
        "Cherry___Peach_disease", "Cherry___Peach_fruit_spot", "Cherry___Peach_iron_deficiency",
        "Cherry___Peach_late_blight", "Cherry___Peach_leaf_curl", "Cherry___Peach_pink_spot",
        "Cherry___Peach_powdery_mildew", "Cherry___Peach_scorch", "Cherry___Peach_spot",
        "Cherry___Peach_stem_end_rot", "Cherry___Peach_white_spot", "Cherry___Peach_yellows",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]  # 38 类，替换为你的实际类别名

    return class_names[predicted_idx.item()], confidence

# 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 预测
    pred_class, confidence = predict_image(filepath)

    # 保存到数据库
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (image_path, predicted_class, confidence) VALUES (?, ?, ?)",
        (filepath, pred_class, confidence)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'class': pred_class,
        'confidence': round(confidence, 4),
        'image_url': f'/static/uploads/{filename}'
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)