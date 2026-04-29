# 手写数字识别 Web 应用

基于 Flask + PyTorch 的手写数字识别 Web 应用，支持实时手写输入和预测。

## 🚀 功能特性

- ✏️ 交互式手写画布（支持鼠标和触屏）
- 🔍 实时数字识别
- 📊 显示 0-9 概率分布
- 🏆 Top-3 预测结果展示
- 📱 响应式设计

## 🛠️ 技术栈

- **框架**: Flask 2.3
- **模型**: PyTorch CNN
- **前端**: HTML5 Canvas + JavaScript

## 📁 项目结构

```
project/
├── app.py              # Web 应用入口（单文件，包含HTML模板）
├── model.pth           # 训练好的模型权重（PyTorch格式）
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

```bash
python app.py
```

### 访问地址

- 本地访问: http://127.0.0.1:5000
- 局域网访问: http://[你的IP]:5000

## 📝 使用方法

1. 在手写区域用鼠标或触屏手写数字（0-9）
2. 数字尽量写大一些，占据画布中心
3. 点击「识别」按钮获取预测结果
4. 点击「清空」按钮重新输入

## 🚀 HuggingFace Spaces 部署

1. 创建新的 Space，选择 **Gradio** 或 **Streamlit**
2. 上传以下文件：
   - `app.py`
   - `model.pth`
   - `requirements.txt`
3. 等待部署完成，获取公网链接

## 📊 模型信息

- 模型类型: PyTorch CNN
- 架构: Conv(32) → Conv(64) → Conv(64) → FC(64) → FC(10)
- 训练准确率: ≥0.98
- 数据集: MNIST

## 📄 许可证

MIT License