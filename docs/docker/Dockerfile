# syntax=docker/dockerfile:1
FROM python:3.10-slim

# 系统依赖（OpenCV、Qt等常见运行库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libxext6 libxrender1 libsm6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件并安装（build context 请使用项目根目录）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 可选：安装CPU版PyTorch（若requirements未指定）
# RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 复制项目
COPY . .

ENV PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1

# 无头运行（xvfb）；如需X11转发，请用 docker run 覆盖 command
CMD ["bash", "-lc", "xvfb-run -a python main.py"]