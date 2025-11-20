<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# YOLOv11 Studio

YOLOv11 Studio 是一款专为计算机视觉工作流设计的专业图像标注工具。它完全在浏览器中运行，并集成了 Google Gemini 模型以实现自动化辅助标注，帮助用户快速创建高质量的 YOLOv11 目标检测数据集。

## 主要功能

- **交互式标注工具**：提供直观的界面进行边界框绘制、选择和调整
- **AI 自动标注**：集成 Google Gemini 2.5 Flash 模型，根据定义的类别自动检测物体
- **类别管理**：支持添加、重命名和自定义类别颜色
- **键盘快捷键**：丰富的快捷键支持，提高标注效率
- **数据集导出**：自动生成标准 YOLOv11 格式数据集，包含训练集/验证集划分
- **训练脚本生成**：自动创建 Python 训练脚本和配置文件
- **数据集导入**：支持导入现有 YOLO 格式数据集进行编辑

## 工具使用说明

### 标注工具与快捷键

| 快捷键 | 功能 | 描述 |
|--------|------|------|
| V | 选择模式 | 选中、移动和调整标注框大小 |
| R | 绘制模式 | 按住鼠标左键拖动以创建新的边界框 |
| H | 平移模式 | 拖动画布以移动视图（在图片放大时非常有用） |
| Del | 删除 | 删除当前选中的标注框 |
| 滚轮 | 缩放 | 放大或缩小画布 |

### 类别管理

在左侧侧边栏管理对象类别（Classes）。您可以添加新类别、修改颜色或重命名。当前高亮的类别将用于新绘制的标注。请确保类别名称具有描述性（例如“猫”、“人”、“红色汽车”），以获得最佳的 AI 自动标注效果。

### AI 自动标注

使用顶部工具栏中的 AI 按钮启动自动标注功能。系统将调用 Gemini 模型分析当前图像，并根据您定义的类别自动检测物体并生成边界框。

### 导出数据集

切换到“训练 / 导出 (Train / Export)”标签页以下载您的数据集。导出的 ZIP 文件包含：

- 训练集和验证集（自动 80/20 划分）
- 标注文件（YOLO 格式）
- data.yaml 配置文件
- Python 训练脚本
- 使用说明文档

## 本地运行

**前置条件：** Node.js

1. 安装依赖：
   `npm install`
2. 在 [.env.local](.env.local) 中设置 `GEMINI_API_KEY` 为您的 Gemini API 密钥
3. 运行应用：
   `npm run dev`

## 技术栈

- React + TypeScript
- Vite 构建工具
- Google Gemini API
- YOLOv11 格式支持
- 响应式设计，支持各种设备

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1WgRMyqbEmm17NF2T0zMCRNYbjIZ-Zi4C



