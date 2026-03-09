# 植物叶片病害识别项目

本项目是一个基于 `PyTorch + Flask` 的植物叶片病害识别系统，包含以下两部分：

- Web 端病害识别与可视化展示
- 基于 PlantVillage 数据集的模型训练、消融实验与结果分析

当前主线入口是 `app.py`，模型结构定义位于 `model/resnet_versions.py`。

## 1. 项目功能

项目主要实现了以下能力：

- 上传植物叶片图片并进行病害识别
- 使用两个 ResNet18 系列模型进行集成预测
- 返回病害名称、植物类别、置信度、严重程度、症状和防治建议
- 保存最近的识别历史记录
- 提供训练脚本、消融实验脚本和结果分析脚本

从当前代码结构看，系统主要围绕 38 类植物叶片健康/病害分类任务构建，数据目录采用 `data/plantvillage/train` 和 `data/plantvillage/val` 的监督学习组织方式。

## 2. 项目结构

```text
pythonProject/
├─ app.py                         # 主 Web 服务，推荐使用
├─ start.py                       # 启动前检查依赖、目录、模型文件
├─ train.py                       # SE 版本 ResNet18 训练脚本
├─ train_ablation.py              # v1/v2 模型对比与消融实验
├─ analyze_results.py             # 训练结果二次分析
├─ generate_paper_figures.py      # 论文图表生成
├─ requirements.txt               # 当前基础依赖
├─ history.json                   # Web 端推理历史记录
├─ model/
│  ├─ resnet_versions.py          # ResNet18_v1 / ResNet18_v2 / ResNet18SE
│  └─ resnet18_se_best.pth        # 主模型权重
├─ results/
│  ├─ resnet18_v1_scratch_best.pth
│  ├─ resnet18_v2_scratch_best.pth
│  └─ *.json / *.csv              # 训练与实验输出
├─ data/
│  └─ plantvillage/
│     ├─ train/
│     ├─ val/
│     └─ test_complex/
├─ templates/
│  └─ index.html                  # 前端页面模板
├─ static/                        # 静态资源
├─ uploads/                       # Web 上传图片目录
├─ figures/                       # 训练图表输出
└─ paper_figures/                 # 论文图输出
```

## 3. 模型说明

项目中主要使用了三种模型定义：

- `ResNet18_v1`：基础版 ResNet18
- `ResNet18_v2`：在 ResNet18 的多个 stage 后加入 `SEBlock`
- `ResNet18SE`：与 `ResNet18_v2` 功能接近，主要用于 `train.py` 训练流程兼容

Web 推理默认尝试加载以下模型：

- `model/resnet18_se_best.pth`
- `results/resnet18_v2_scratch_best.pth`

如果模型加载失败，`app.py` 中还保留了后备识别逻辑，但实际部署建议优先保证模型文件存在。

## 4. 运行环境

建议环境：

- Python 3.9 及以上
- Windows
- 可选 CUDA 环境（有 GPU 时会自动使用）

安装基础依赖：

```bash
pip install -r requirements.txt
```

注意：当前 `requirements.txt` 只覆盖了 Web 推理的核心依赖。若你要运行训练、分析和绘图脚本，还需要额外安装这些常用包：

```bash
pip install tqdm matplotlib seaborn scikit-learn
```

## 5. 快速启动 Web 系统

### 方式一：直接启动主程序

```bash
python app.py
```

启动后默认访问：

```text
http://127.0.0.1:5000
```

### 方式二：使用启动检查脚本

```bash
python start.py
```

`start.py` 会先检查：

- 依赖是否已安装
- 模型文件是否存在
- `uploads`、`templates`、`static`、`model` 等目录是否存在

## 6. 数据集准备

训练脚本默认读取以下目录：

```text
data/plantvillage/train
data/plantvillage/val
```

也就是说，数据需要按 `ImageFolder` 规范组织，例如：

```text
data/plantvillage/train/Apple___Apple_scab/*.jpg
data/plantvillage/train/Apple___healthy/*.jpg
data/plantvillage/val/Tomato___Late_blight/*.jpg
```

当前仓库中已经存在：

- `data/plantvillage/train`
- `data/plantvillage/val`
- `data/plantvillage/test_complex`

## 7. 模型训练

### 7.1 训练主模型

运行：

```bash
python train.py
```

该脚本会：

- 使用 `ResNet18SE`
- 从 `data/plantvillage/train` / `val` 读取数据
- 将最佳模型保存到 `model/resnet18_se_best.pth`
- 将训练曲线、混淆矩阵、分类 F1 图等输出到 `figures/`

### 7.2 运行消融实验

运行：

```bash
python train_ablation.py
```

该脚本会对比：

- `ResNet18_v1`
- `ResNet18_v2 (+SE)`

输出包括：

- 最优权重文件到 `results/`
- 训练历史 `*_history.json`
- 指标对比图和混淆矩阵图

## 8. 结果分析与论文图表

### 分析已有训练结果

```bash
python analyze_results.py
```

适合在已有 `results/*.pth` 和 `results/*_history.json` 的情况下进行二次评估。

### 生成论文图表

```bash
python generate_paper_figures.py
```

输出目录为：

```text
paper_figures/
```

## 9. Web 接口说明

主服务在 `app.py` 中提供了以下接口：

- `GET /`：主页
- `POST /api/predict`：上传图片并返回识别结果
- `GET /api/history`：获取识别历史
- `POST /api/history/clear`：清空识别历史
- `GET /api/statistics`：获取统计信息
- `GET /api/diseases`：获取病害列表

其中 `POST /api/predict` 需要上传表单文件字段：

```text
file
```

返回内容一般包含：

- 病害名称
- 植物名称
- 严重程度
- 置信度
- 症状说明
- 防治建议
- 热力图结果
- 历史记录信息

## 10. 输出文件说明

- `uploads/`：Web 端用户上传的图片
- `history.json`：最近 50 条识别记录
- `figures/`：主训练脚本输出图像
- `results/`：消融实验模型和指标文件
- `paper_figures/`：论文绘图输出

## 11. 备注

- `app.py` 是当前更完整的 Web 版本，建议优先使用。
- `main.py` 也是一个 Flask 程序，但从当前仓库结构看更像早期版本或备用实现，不建议作为默认入口。
- 当前代码中部分终端输出存在编码显示异常，但不影响你理解主流程；如果需要，我可以继续帮你把这些中文注释和日志统一修正为正常编码。

## 12. 常用命令汇总

```bash
# 安装依赖
pip install -r requirements.txt
pip install tqdm matplotlib seaborn scikit-learn

# 启动 Web 服务
python app.py

# 启动前检查
python start.py

# 训练主模型
python train.py

# 运行消融实验
python train_ablation.py

# 分析结果
python analyze_results.py

# 生成论文图表
python generate_paper_figures.py
```
