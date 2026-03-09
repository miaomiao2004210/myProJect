"""
快速启动脚本 - 自动检查环境并启动系统
"""
import os
import sys
import subprocess

def check_dependencies():
    """检查依赖包"""
    print("检查依赖包...")
    
    required_packages = {
        'flask': 'Flask',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'torch': 'torch',
        'torchvision': 'torchvision'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_model():
    """检查模型文件"""
    print("\n检查模型文件...")
    
    model_path = 'model/resnet18_se_best.pth'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ 模型文件存在: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ✗ 模型文件不存在: {model_path}")
        print("  提示: 请确保模型文件在正确位置")
        return False

def check_directories():
    """检查必要目录"""
    print("\n检查目录结构...")
    
    required_dirs = ['uploads', 'templates', 'static', 'model']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (不存在)")
            os.makedirs(dir_name)
            print(f"    已创建: {dir_name}/")
    
    return True

def start_app():
    """启动应用"""
    print("\n" + "=" * 60)
    print("启动植物叶片病害识别系统...")
    print("=" * 60)
    
    try:
        # 导入并运行app
        import app
        # app.py会自动运行
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "植物叶片病害识别系统启动器" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型
    model_exists = check_model()
    if not model_exists:
        print("\n警告: 模型文件不存在，系统将使用后备识别方案")
        response = input("是否继续启动? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 检查目录
    check_directories()
    
    # 启动应用
    start_app()

if __name__ == '__main__':
    main()
