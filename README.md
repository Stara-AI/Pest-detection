# 农作物哨兵系统手册（用户与运维）

博客：[[基于YOLOv8-SEAttention与LLMs融合的农作物害虫智能诊断与防控决策系统-CSDN博客](https://blog.csdn.net/m0_69402477/article/details/150581782?spm=1001.2014.3001.5501)](https://blog.csdn.net/m0_69402477/article/details/150581782?spm=1001.2014.3001.5501)

## 0. 版本信息
- 文档版本：v1.0
- 适用代码基线：本仓库当前提交
- 更新日期：以仓库提交时间为准

## 1. 项目概述
CropGuard 是一套基于 Ultralytics YOLOv8 的农作物病虫害智能检测与防治建议系统，提供桌面端图形界面（PyQt5）、视频/图像/摄像头多源输入、检测可视化与一键生成 AI 防治方案。系统内置 SEAttention（Squeeze-and-Excitation）注意力机制可选模型配置，同时封装了多家中文大模型 API（智谱、阿里千问、百度千帆、豆包、DeepSeek、OpenAI 兼容）用于生成结构化的农技建议。

核心能力：
- 实时/离线检测：图片、目录批处理、视频逐帧、摄像头流
- 可视化与归档：检测框渲染、中文类名映射、结果图与文本方案按类归档
- AI 建议：根据检测结果自动生成 1500 字左右综合防治技术建议
- 可配置：通过 `config/configs.yaml` 配置模型、UI、设备、AI 提示词与 API 等
- 跨平台与容器化：Windows/Linux 本地运行，提供 CPU/GPU Docker 镜像与 Compose

## 2. 目录结构与关键文件
```
Pest-detection/
├─ main.py                 # GUI 主程序与推理、AI 建议逻辑
├─ UI.py                   # 由 Qt Designer 生成的界面类
├─ config/
│  ├─ configs.yaml         # 系统配置（模型/推理/UI/AI/通用）
│  └─ traindata.yaml       # 训练/验证/测试数据集与类别定义
├─ tool/
│  ├─ parser.py            # YAML 配置解析器（EasyDict）
│  └─ tools.py             # 绘制、结果格式化、导出与图像填充等公用函数
├─ prompts/
│  ├─ core/prompt_manager.py  # 提示词加载与统一访问接口
│  └─ templates/*.txt         # 各模型提示词模板
├─ ultralytics/            # 框架源码（含自定义 SEAttention 与模型 YAML）
│  └─ cfg/models/v8/det_self/yolov8s-attention-SE.yaml
├─ weights/                # 训练权重与训练记录
├─ docs/docker/            # Dockerfile 与 docker-compose.yml
├─ dataset/                # 数据集（train/val/test, YOLO 标签）
└─ output/                 # 保存的检测结果与 AI 文本方案（按类名归档）
```

## 3. SENet网络结构

`SE`块是一种创新性架构单元，它通过动态调整通道特征来增强网络的表征能力。实验证明，`SENet`在多个数据集和任务上都取得了领先的性能表现。该设计还揭示了传统架构在建模通道特征依赖关系方面的局限性。`SE`块的这一特性有望拓展到其他需要高区分度特征的任务中。此外，`SE`块生成的特征重要性指标还可应用于模型压缩等场景，如网络剪枝。

> `论文地址`：[https://arxiv.org/pdf/1709.01507.pdf](https://arxiv.org/pdf/1709.01507.pdf)
> `代码地址`：[https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/522309b123564587b2d9cc0614f4367a.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c5c9c7fa6164deea9675c5cccab5c99.png)

### 3.1 YOLOV8中集成SEAttention

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e120dcc5f7f147ddbe085469bdca5d10.png)

处注意修改层数的变化，层数是从**0**开始数的，由于此处是添加到了第**10**层，因此后面层数都发生了变化。**10**层以后的相关层数都需要加**1**。具体修改内容如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31f7835db6d341998ab8145d17155da9.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d4cc77ef8574c3bbbeac039bdfd49c8.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/296f4332ecf743aaab6647b66341aa46.png)

1. **数据集**

   本数据集共包含 **18976** 张图像，涵盖了 **102** 类常见农作物害虫种类，覆盖水稻、小麦、玉米、棉花、果树及其他经济作物中高发的害虫类别。每一类图片均标注了对应虫害名称，适用于图像分类、目标检测及深度学习任务。
   数据集涵盖的虫害包括但不限于：
   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/162c7f3082ef41188d1175bdeb5c3277.png)

    - **水稻类害虫**：稻纵卷叶螟、稻螟蛉、稻飞虱、稻蓟马、稻瘿蚊、稻水象甲等；
    - **小麦类害虫**：麦蜘蛛、麦蚜、麦叶甲、小麦吸浆虫等；
    - **玉米类害虫**：玉米螟、粘虫、蚜虫等；
    - **棉花、豆类及果树类害虫**：红蜘蛛、盲蝽象、蓟马、介壳虫、实蝇、木蠹蛾、潜叶蛾等；
    - **广义农业害虫**：斑潜蝇、跳甲、象鼻虫、绿盲蝽、地老虎、灰象甲等；
    - **外来入侵害虫与区域性高发种类**：荔枝蝽、黄脊竹蝗、美国白蛾、落叶卷叶蛾、中华稻蝗等。 

   | 数据集 | 图片总数 | 标注框总数 |
   | ------ | -------- | ---------- |
   | train  | 15180    | 17791      |
   | val    | 1897     | 2230       |
   | test   | 1899     | 2263       |
   | 总计   | 18976    | 22284      |

## 4. 用户界面设计

基于`PyQt5`的现代化`GUI`界面，支持：

- 多源输入管理（图片/视频/摄像头/目录）
- 实时检测结果可视化
- 检测参数动态调整
- `AI`建议异步获取与展示
- 结果导出与归档管理
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ffae7aebb1b343d9a94e2faafceea9d0.png)

### 4.1 运行时架构与数据流

- UI 层：`MyMainWindow(QMainWindow, Ui_MainWindow)` 负责交互、显示、路径选择、定时帧处理（QTimer 20ms）。
- 推理层：使用 `YOLO(weights).predict(img, imgsz, conf, device, classes)` 返回框、类别与置信度；`tool.tools.format_data` 统一为 `[name, score, [x1,y1,x2,y2]]`。
- 可视化：`tool.tools.draw_info` 渲染检测框与标签；`resize_with_padding` 保持纵横比填充显示。
- AI 决策：`AIClient` 采用策略模式封装多家 API；`AdviceWorker(QThread)` 异步拉取文本建议；`prompts.core.prompt_manager` 负责模板加载。
- 输出归档：用户点击“保存结果”后，系统在 `output/` 下以中文类别名合成目录，`类别名.JPG` 与`类别名防治方案.txt`。
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9070fa33e89f47858d0a871b62a02fc9.png)
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/597c19a6b9af4a2c8fdb0ce5481d15c3.png)
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b0d7e6e49474ac7a85046c4123fba69.png)
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aa03f7dbe6b44e038fabc2357bb667ab.png)

## 5. 运行时架构与数据流

- UI 层：`MyMainWindow(QMainWindow, Ui_MainWindow)` 负责交互、显示、路径选择、定时帧处理（QTimer 20ms）。
- 推理层：使用 `YOLO(weights).predict(img, imgsz, conf, device, classes)` 返回框、类别与置信度；`tool.tools.format_data` 统一为 `[name, score, [x1,y1,x2,y2]]`。
- 可视化：`tool.tools.draw_info` 渲染检测框与标签；`resize_with_padding` 保持纵横比填充显示。
- AI 决策：`AIClient` 采用策略模式封装多家 API；`AdviceWorker(QThread)` 异步拉取文本建议；`prompts.core.prompt_manager` 负责模板加载。
- 输出归档：用户点击“保存结果”后，系统在 `output/` 下以中文类别名合成目录，保存结果 JPG 与“防治方案.txt”。

## 6. 安装与部署
### 6.1 环境要求
- 操作系统：Windows 10/11（推荐）、Ubuntu 20.04+；macOS 可运行但 PyQt/显示依赖需额外处理
- Python：3.8～3.10（建议 3.10）
- GPU（可选）：NVIDIA 显卡与匹配 CUDA 驱动

### 6.2 依赖安装（本地）

以下为分平台的完整指令。建议先准备虚拟环境并升级 pip。

#### 6.2.1 Windows（PowerShell）
```bash
# 1) 创建与激活虚拟环境
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip

# 2) 安装依赖（优先使用项目内 requirements）
pip install -r requirements.txt

# 2.1) 如需CPU版PyTorch（若requirements未固定或需替换）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2.2) 如需GPU版PyTorch（示例：CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) 首次运行（GUI）
python main.py
```

（CMD 命令差异）
```bat
:: CMD环境激活
.venv\Scripts\activate
python -m pip install --upgrade pip
```

#### 6.2.2 Ubuntu/Debian（apt）
```bash
# 0) 系统依赖（OpenCV/Qt运行库等）
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  build-essential git curl ca-certificates \
  libgl1 libglib2.0-0 libxext6 libxrender1 libsm6

# 1) 虚拟环境
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) 安装依赖
pip install -r requirements.txt
# 可选：CPU/GPU PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) 首次运行（无头服务器可参考4.3）
python main.py
```

#### 6.2.3 CentOS/RHEL（yum/dnf）
```bash
# 0) 系统依赖
sudo yum -y update || sudo dnf -y update
sudo yum install -y \
  python3 python3-pip python3-venv \
  gcc gcc-c++ make git curl \
  mesa-libGL mesa-libGLU libXext libXrender libSM

# 1) 虚拟环境
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) 安装依赖
pip install -r requirements.txt
# 可选：CPU/GPU PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) 运行
python main.py
```

#### 6.2.4 Ubuntu/CentOS
```bash
# 系统依赖（Ubuntu 示例）
sudo apt-get update && sudo apt-get install -y \
  python3 python3-pip python3-venv build-essential \
  libgl1-mesa-glx libglib2.0-0 libxext6 libxrender1 libsm6 \
  libxrandr2 libasound2 libgtk-3-0 libgstreamer1.0-0 \
  libgstreamer-plugins-base1.0-0

# Python 虚拟环境
python3 -m venv pest-env
source pest-env/bin/activate
pip install --upgrade pip

# 获取代码并安装
git clone https://github.com/your-repo/pest-detection.git
cd pest-detection
pip install -r requirements.txt

# 可选：PyQt5 安装问题
sudo apt-get install -y python3-pyqt5 || pip install PyQt5==5.15.9

# 运行
python main.py
```

#### 6.2.5 macOS
```bash
# Homebrew 安装
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python3 qt5 opencv
export PATH="/usr/local/opt/qt5/bin:$PATH"

python3 -m venv pest-env
source pest-env/bin/activate

git clone https://github.com/your-repo/pest-detection.git
cd pest-detection
pip install -r requirements.txt
```

### 6.3 服务器（无头）部署

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv xvfb

python3 -m venv pest-env
source pest-env/bin/activate

# 获取项目与依赖
git clone https://github.com/your-repo/pest-detection.git
cd pest-detection
pip install -r requirements.txt
pip install opencv-python-headless

# 启动脚本
cat > start_pest_detection.sh << 'EOF'
#!/bin/bash
cd /path/to/pest-detection
source pest-env/bin/activate
export QT_X11_NO_MITSHM=1
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
python main.py
EOF
chmod +x start_pest_detection.sh

# 后台运行（screen/tmux 二选一）
screen -S pest-detection -dm bash -lc './start_pest_detection.sh'
```

#### 6.3.1 systemd 服务
```bash
sudo tee /etc/systemd/system/pest-detection.service << EOF
[Unit]
Description=Pest Detection System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/pest-detection
Environment=PATH=/home/ubuntu/pest-detection/pest-env/bin
Environment=DISPLAY=:99
ExecStartPre=/usr/bin/Xvfb :99 -screen 0 1024x768x24
ExecStart=/home/ubuntu/pest-detection/pest-env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now pest-detection
```

### 6.4 Docker 部署（docs/docker）

#### 6.4.1 CPU 镜像
```bash
# 构建（在项目根目录）
docker build -f docs/docker/Dockerfile -t pest-detection:cpu .
# 运行
docker run -it --rm --name pest-cpu -v $(pwd):/app pest-detection:cpu
```

#### 6.4.2 GPU 镜像（CUDA 11.8）
```bash
# 安装 nvidia-docker2 并重启 docker 后：
docker build -f docs/docker/Dockerfile.gpu -t pest-detection:gpu .
docker run --rm --gpus all pest-detection:gpu nvidia-smi

docker run -it --rm --name pest-gpu --gpus all -v $(pwd):/app pest-detection:gpu
```

#### 6.4.3 Docker Compose（CPU/GPU）
```bash
cd docs/docker
# 启动 CPU
docker-compose up pest-cpu -d
# 启动 GPU
docker-compose up pest-gpu -d
# 查看状态
docker-compose ps
# 查看日志
docker-compose logs -f pest-cpu
```

## 7. 配置说明（config/configs.yaml）

- `MODEL.WEIGHT`：推理权重路径，如 `./weights/yolov8s/weights/best.pt`
- `MODEL.DEVICE`：`cpu` 或 `0/1/2...` 指定 GPU 编号
- `MODEL.CONF`：置信度阈值（默认 0.4）
- `OUTPUT.*`：结果输出目录/格式
- `UI.*`：界面主题色、列宽、背景图
- `CONFIG.chinese_name`：类别到中文名映射（完整见文件）
- `AI.active_model` 与各模型密钥、超时、温度、max_tokens 等

## 8. 故障排除

- 摄像头无画面：检查占用；修改 `camera_num`；驱动问题
- 模型加载失败：确认权重路径；PyTorch/CUDA 版本匹配
- 大模型调用失败：网络/API Key/证书与超时设置
- UI 显示异常：PyQt5 版本、字体/资源路径

## 9. 优化建议

- 推理：ONNX/TensorRT 导出与量化；批处理
- 可靠性：systemd 自启动、日志轮转、异常恢复
- 安全：Docker 网络隔离、最小权限、基础镜像更新

## 10. 二次开发指引

- 新类别：补充数据-标注-训练，更新 `traindata.yaml` 与 `configs.yaml`
- 新 AI 提供商：新增模板（`prompts/templates`）与 `AIClient` 分支
- UI 定制：通过 `UI.ui`（Qt Designer）调整并 `pyuic5` 生成 `UI.py`

---

**项目名称**：农作物哨兵  
**技术架构**：YOLOv8 + SE 注意力机制 + 多模型防控建议  
**适用场景**：农业有害生物快速识别、巡检监测、专家化建议  
**版本**：v1.0  
**更新时间**：2025-08 
