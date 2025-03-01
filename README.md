# 课件内容提取与优化工具 (PPT_Text_Extractor)

## 简介
课件内容提取与优化工具是一个基于 Python 的智能工具，旨在从 PPT 文件中提取文本和图片内容，并通过优化处理生成自然流畅的文本输出。项目利用 `python-pptx` 和 `PaddleOCR` 技术，支持幻灯片文本提取和图片文字识别，适合教学文档整理或后续自动化处理。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Version](https://img.shields.io/badge/version-1.0.0-yellow.svg)

---

## 功能特点
- 📄 **PPT 文本提取**：自动提取幻灯片的标题、段落和表格内容。
- 🖼️ **图片文本识别**：使用 PaddleOCR 从幻灯片图片中提取文字。
- ✍️ **文本优化**：将提取的内容优化为叙述性文本，便于阅读或后续使用。
- 📋 **结构化输出**：按幻灯片分隔保存文本和图片内容。
- ⚙️ **日志记录**：提供详细的处理日志，便于调试。

---

## 技术栈
- **Python**: 3.8 或更高版本
- **python-pptx**: 解析 PPTX 文件
- **PaddleOCR**: 图片文本识别
- **logging**: 日志记录

---

## 安装步骤

### 前置条件
- Python 3.8 或以上版本
- Git（用于克隆仓库）
- 可选：GPU 支持（加速 PaddleOCR）

### 安装依赖
```bash
# 1. 克隆项目仓库
git clone https://github.com/blankboards/pptx_extraction.git
cd pptx_extraction

# 2. 创建并激活虚拟环境
python -m venv venv
# Unix/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖列表 (requirements.txt 示例)
```
python-pptx>=0.6.21
paddlepaddle>=2.5.0
paddleocr>=2.7.0
```

---

## 快速开始

### 基本使用
以下是如何提取和优化 PPT 内容的示例：
```python
from main import process_ppt_file

# 指定 PPT 文件路径
ppt_path = 'sample.pptx'

# 处理 PPT 并生成优化文本
process_ppt_file(ppt_path)
print("优化文本已保存至 output/optimized_output.txt")
```

### 输出示例
```
Title: N/A
Author: huawei

现在我们来看看第 1 张幻灯片：
首先是 深度学习基础。
这里有个重点 本课程将深入讲解深度学习的基本原理。
```

---

## 配置参数

### 主函数参数
| 参数名       | 类型  | 默认值        | 描述                     |
|--------------|-------|---------------|--------------------------|
| `file_path`  | `str` | 必填          | PPT 文件路径             |
| `output_dir` | `str` | `'output'`    | 输出目录（定义在 config.py） |

### 可调整配置 (config.py)
- `OUTPUT_DIR`: 输出文本和图片的目录
- `PPTX_FILE`: 默认处理的 PPT 文件路径

---

## 项目结构
```
PPT_Text_Extractor/
├── modules/                 # 功能模块目录
│   ├── __init__.py          # 包初始化文件
│   ├── ppt_text_extraction.py  # PPT 文本提取模块
│   ├── image_extraction_p.py   # 图片文本提取模块
│   ├── ai_optimizer.py         # 文本优化模块
│   ├── utils.py                # 工具函数
│   └── config.py               # 配置文件
├── output/                  # 输出目录（运行时生成）
│   ├── slide_1/             # 幻灯片 1 的输出
│   │   ├── image/           # 图片文件
│   │   └── slide_1_texts.txt  # 图片文本
│   └── optimized_output.txt # 优化后的文本
├── logs/                    # 日志目录（运行时生成）
│   └── process.log          # 处理日志
├── main.py                  # 主入口脚本
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明文档
```

---

## 使用指南

### 运行步骤
1. 准备一个 `.pptx` 文件（如 `sample.pptx`）。
2. 在 `main.py` 中设置 `PPTX_FILE_2` 为你的文件路径。
3. 运行：
   ```bash
   python main.py
   ```
4. 查看 `output/optimized_output.txt` 中的结果。

### 自定义输出
- 修改 `ai_optimizer.py` 调整文本优化逻辑。
- 在 `config.py` 中更改 `OUTPUT_DIR` 设置输出路径。

---

## 常见问题 (FAQ)

1. **支持哪些文件格式？**
   - 目前支持 `.pptx` 格式。
   - 不支持 `.ppt`，建议转换为 `.pptx`。

2. **图片文本识别不准确怎么办？**
   - 确保图片清晰，调整 PaddleOCR 的置信度阈值。

3. **处理速度慢怎么办？**
   - 启用 GPU（安装 paddlepaddle-gpu）。
   - 减少幻灯片中的图片数量。

4. **依赖安装失败怎么办？**
   - 检查网络，运行 `pip install --upgrade pip`。
   - 确保 Python 版本 >= 3.8。

---

## 许可证
本项目采用 [MIT License](https://opensource.org/licenses/MIT)，欢迎使用和修改。

---

## 贡献指南
欢迎贡献代码或建议！
- **报告问题**：提交 GitHub Issue。
- **提交代码**：Fork 仓库并创建 Pull Request。
- **规范**：遵循 PEP 8，添加注释。

### 贡献流程
1. Fork 仓库。
2. 创建分支（`git checkout -b feature/xxx`）。
3. 提交更改（`git commit -m "描述"`）。
4. Push 到远程（`git push origin feature/xxx`）。
5. 创建 Pull Request。

---

## 联系方式
- **邮箱**：zhanghoubing777@gmail.com
- **GitHub**：https://github.com/BlairCode
