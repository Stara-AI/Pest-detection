import traceback
from datetime import datetime
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor

from UI import Ui_MainWindow

from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog

import cv2
import os
import numpy as np
import shutil

from ultralytics import YOLO

from tool.parser import get_config
from tool.tools import draw_info, result_info_format, format_data, writexls, writecsv, resize_with_padding
from prompts.core.prompt_manager import prompt_manager

import winsound
import requests
import json

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 从配置文件加载全局变量
config = get_config('./config/configs.yaml')
# UI配置
title = config.get('UI', {}).get('title', '农作物虫害检测识别系统')
label_title = config.get('UI', {}).get('label_title', '农作物虫害检测识别系统')
background = config.get('UI', {}).get('background', './icon/background.jpg')
zhutu2 = config.get('UI', {}).get('zhutu2', './icon/zhutu2.png')
# 摄像头配置
camera_num = config.get('CONFIG', {}).get('camera_num', 0)
# 中文名称映射
chinese_name = config.get('CONFIG', {}).get('chinese_name', {})

# AI配置
"""......."""

# AI模型显示名称映射
"""........"""

"""class AIClient"""
    """AI客户端，支持多种模型API"""

    def __init__(self, model_config):
        self.provider = model_config.get('provider', 'openai')
        self.api_base = model_config.get('api_base', '')
        self.model = model_config.get('model', '')
        self.api_key = model_config.get('api_key', '')
        self.secret_key = model_config.get('secret_key', '')
        self.verify_ssl = model_config.get('verify_ssl', True)

    def get_advice(self, pest_info, timeout=30):
        """获取防治建议"""
        try:
            if self.provider == 'qianfan':
                return self._call_qianfan_api(pest_info, timeout)
            elif self.provider == 'doubao':
                return self._call_doubao_api(pest_info, timeout)
            else:
                return self._call_openai_compatible_api(pest_info, timeout)
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")

    def _call_openai_compatible_api(self, pest_info, timeout):
        """调用OpenAI兼容的API"""
        # 智谱AI需要特殊的处理
        if self.provider == "zhipu":
            # 智谱AI的api_base已经包含完整路径
            api_url = self.api_base
            # 智谱AI需要特殊的Authorization头格式
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 使用提示词管理器获取智谱AI的提示词
            system_content = prompt_manager.get_prompt('zhipu')
        elif self.provider == "qwen":
            api_url = f"{self.api_base.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 使用提示词管理器获取千问的提示词
            system_content = prompt_manager.get_prompt('qwen')
        elif self.provider == "deepseek":
            api_url = f"{self.api_base.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 使用提示词管理器获取DeepSeek的提示词
            system_content = prompt_manager.get_prompt('deepseek')
        elif self.provider == "openai":
            api_url = f"{self.api_base.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 使用提示词管理器获取OpenAI的提示词
            system_content = prompt_manager.get_prompt('openai')
        else:
            api_url = f"{self.api_base.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 使用提示词管理器获取默认提示词
            system_content = prompt_manager.get_prompt('default')

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": pest_info}
            ],
            "max_tokens": ai_max_tokens,
            "temperature": ai_temperature,
            "stream": False  # 关闭流式响应，确保完整返回
        }

        # 禁用代理设置，避免连接问题
        proxies = {
            "http": None,
            "https": None
        }

        try:
            # print(f"正在请求 {self.provider} API，超时时间: {timeout}秒")  # 隐藏API请求信息
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=self.verify_ssl,
                proxies=proxies
            )
        except requests.exceptions.Timeout:
            raise Exception(f"请求超时（{timeout}秒），请检查网络连接")
        except Exception as e:
            raise Exception(f"请求失败: {str(e)}")

        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raise Exception(f"API调用失败（状态码：{response.status_code}）")

    def _call_qianfan_api(self, pest_info, timeout):
        """调用百度千帆大模型API"""
       

        try:
            # print(f"正在请求百度千帆大模型API，超时时间: {timeout}秒")  # 隐藏API请求信息
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=self.verify_ssl,
                proxies=proxies
            )
        except requests.exceptions.Timeout:
            raise Exception(f"请求超时（{timeout}秒），请检查网络连接")
        except Exception as e:
            raise Exception(f"请求失败: {str(e)}")

        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raise Exception(f"百度千帆大模型API调用失败（状态码：{response.status_code}）")

    def _call_doubao_api(self, pest_info, timeout):
        """调用豆包大模型API"""
        # 豆包大模型使用OpenAI兼容的API格式
        api_url = f"{self.api_base.rstrip('/')}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt_manager.get_prompt('doubao')},
                {"role": "user", "content": pest_info}
            ],
            "max_tokens": ai_max_tokens,
            "temperature": ai_temperature,
            "stream": False  # 关闭流式响应，确保完整返回
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 禁用代理设置，避免连接问题
        proxies = {
            "http": None,
            "https": None
        }

        try:
            # print(f"正在请求豆包大模型API，超时时间: {timeout}秒")  # 隐藏API请求信息
           

class AdviceWorker(QThread):
    """AI建议获取工作线程"""
    success = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ai_client, pest_info, timeout):
        super().__init__()
        self.ai_client = ai_client
        self.pest_info = pest_info
        self.timeout = timeout

    def run(self):
        try:
            advice = self.ai_client.get_advice(self.pest_info, self.timeout)
            self.success.emit(advice)
        except Exception as e:
            self.error.emit(str(e))


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, cfg=None):
        super().__init__()
        self.frame_number = 0
        self.comboBox_index = None
        self.results = []
        self.result_img_name = None
        self.setupUi(self)

        # 根据config配置文件更新界面配置
        self.init_UI_config()
        self.start_type = None
        self.img = None
        self.img_path = None
        s
        self.result_session_dir = None

        # 默认选择为所有目标
        self.comboBox_value = '所有目标'

        self.number = 1
        self.RowLength = 0
        self.consum_time = 0
        self.input_time = 0

        # 打开图片
        self.pushButton_img.clicked.connect(self.open_img)

        # 异步建议线程
        self.advice_thread = None
        # 打开文件夹
        self.pushButton_dir.clicked.connect(self.open_dir)
        # 打开视频
        self.pushButton_video.clicked.connect(self.open_video)
        # 打开摄像头
        self.pushButton_camera.clicked.connect(self.open_camera)
        # 绑定开始运行
        self.pushButton_start.clicked.connect(self.start)
        # 导出数据
        self.pushButton_export.clicked.connect(self.write_files)
        
        # 保存结果
        self.pushButton_save.clicked.connect(self.save_current_result)

        # 获取防治建议
        self.pushButton_advice.clicked.connect(self.get_advice)

        self.comboBox.activated.connect(self.onComboBoxActivated)
        self.comboBox.mousePressEvent = self.handle_mouse_press

        # 表格点击事件绑定
        self.tableWidget_info.cellClicked.connect(self.cell_clicked)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_files = []
        self.current_index = 0
        # 运行状态标志，避免依赖按钮文字
        self.is_running = False

    def init_UI_config(self):
        """
        根据config.yaml中的配置，更新界面
        """
        # 更新界面标题
        self.setWindowTitle(title)
        # 更新 label_title 的标题文本
        self.label_title.setText(label_title)
        # 更新背景图片
        self.setStyleSheet("#centralwidget {background-image: url('%s')}" % background_img)
        # 更新主图
        self.label_img.setPixmap(QtGui.QPixmap(zhutu2))
        # # 更新姓名学号等信息
        # self.label_info.setText(label_info_txt)
        # self.label_info.setStyleSheet("color: rgb(%s);" % label_info_color)
        # self.pushButton_start.setStyleSheet(
        #     "background-color: rgb(%s); border-radius: 15px; color: rgb(%s); " % (start_button_bg, start_button_font))
        # self.pushButton_export.setStyleSheet(
        #     "background-color: rgb(%s); border-radius: 15px; color: rgb(%s);" % (export_button_bg, export_button_font))
        # 左侧控制区域颜色
        self.label_control.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_control_color)
        self.label_img.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_img_color)
        # 设置 tableWidget 表头 的样式
        header_style_sheet = """
                    QHeaderView::section {{
                        background-color: rgb({header_background_color});
                        color: {header_color};
                    }}
                    """.format(
            header_background_color=header_background_color,
            header_color=header_color
        )
        self.tableWidget_info.horizontalHeader().setStyleSheet(header_style_sheet)

        
    def cell_clicked(self, row, column):
        """
        列表 单元格点击事件
        """
        self.update_comboBox_default()

        result_info = {}
        # 判断此行是否有值
        if self.tableWidget_info.item(row, 1) is None:
            return

        # 图片路径
        self.img_path = self.tableWidget_info.item(row, 1).text()
        # 识别结果
        self.results = eval(self.tableWidget_info.item(row, 3).text())
        # 保存路径
        self.result_img_name = self.tableWidget_info.item(row, 6).text()

        # 如果有已保存图片则读取，否则用原图+当前结果重绘
        try:
            if self.result_img_name and os.path.exists(self.result_img_name):
                self.img_show = cv2.imdecode(np.fromfile(self.result_img_name, dtype=np.uint8), -1)
            else:
                src_img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.img_show = draw_info(src_img, self.results)
        except Exception:
            self.img_show = None

        if len(self.results) > 0:
            box = self.results[0][2]
            score = self.results[0][1]
        
    def onComboBoxActivated(self):
        """
        点击下拉列表
        """
        self.sign = True
        # 选择的值
        comboBox_text = self.comboBox.currentText()
        # 值对应的索引
        self.comboBox_index = self.comboBox.currentIndex()
        result_info = {}

        if len(self.results) == 0:
            print('图片中无目标！')
            QMessageBox.information(self, "信息", "图片中无目标", QMessageBox.Yes)
            return
        # 所有目标，默认显示结果中的第一个
        if comboBox_text == '所有目标':
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
            lst_info = self.results
        else:
            # 通过索引确定选择的目标对象
            select_result = self.results[self.comboBox_index - 1]
            box = select_result[2]
            cls_name = select_result[0]
            score = select_result[1]
            lst_info = [[cls_name, score, box]]

        # 格式拼接
        result_info = result_info_format(result_info, box, score, cls_name)

        self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.img_show = draw_info(self.img, lst_info)
        self.show_all(self.img_show, result_info)

    def show_frame(self, img):
        self.update()
        if img is not None:
            
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件  ;;All Files (*)
            self.img_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                  "JPEG Image (*.jpg);;PNG Image (*.png);;JFIF Image (*.jfif)")
            if self.img_path == "":  # 未选择文件
                self.start_type = None
                return

            self.img_name = os.path.basename(self.img_path)
            # 显示相对应的文字
            self.label_img_path.setText(" " + self.img_path)
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.start_type = 'img'
            # 读取中文路径下图片
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 显示原图
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_dir(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            self.img_path_dir = QFileDialog.getExistingDirectory(None, "选择文件夹")
            if not self.img_path_dir:
                self.start_type = None
                return

            self.start_type = 'dir'
            # 显示相对应的文字
            self.label_dir_path.setText(" " + self.img_path_dir)
            self.label_img_path.setText(" 选择图片文件")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.image_files = [file for file in os.listdir(self.img_path_dir) if file.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

            if not self.image_files:
                QMessageBox.information(self, "信息", "文件夹中没有符合条件的图片", QMessageBox.Yes)
                return

            self.current_index = 0
            self.img_path = os.path.join(self.img_path_dir, self.image_files[self.current_index])
            self.img_name = self.image_files[self.current_index]

            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_video(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件
            self.video_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                    "mp4 Video (*.mp4);;avi Video (*.avi)")
            if not self.video_path:
                self.start_type = None
                return

            self.start_type = 'video'
            # 显示相对应的文字
            self.label_video_path.setText(" " + self.video_path)
            self.label_img_path.setText(" 选择图片文件")
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_camera_path.setText(" 打开摄像头")

            self.video_name = os.path.basename(self.video_path)
            self.video = cv2.VideoCapture(self.video_path)
            # 读取第一帧
            ret, self.img = self.video.read()
            if ret:
                # 设置图像名称
                self.img_name = f"{self.video_name}_frame.jpg"
                self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_camera(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            if self.label_camera_path.text() == ' 打开摄像头' or self.label_camera_path.text() == ' 摄像头已关闭':
                self.start_type = 'camera'
                # 显示相对应的文字
                self.label_img_path.setText(" 选择图片文件")
                self.label_dir_path.setText(" 选择图片文件夹")
                self.label_video_path.setText(" 选择视频文件")
                self.label_camera_path.setText(" 摄像头已打开")

                self.video_name = camera_num
                self.video = cv2.VideoCapture(self.video_name)
                ret, self.img = self.video.read()
                if ret:
                    # 设置图像名称
                    self.img_name = "camera_frame.jpg"
                    self.show_frame(self.img)
            elif self.label_camera_path.text() == ' 摄像头已打开':
                # 修改文本为开始运行
                self.pushButton_start.setText("开始运行 >")
                self.label_camera_path.setText(" 摄像头已关闭")

        except Exception as e:
            traceback.print_exc()

    def show_all(self, img, info):
        '''
        展示所有的信息
        '''
        print(f"Debug: show_all called with img shape: {img.shape if img is not None else 'None'}")
        print(f"Debug: info: {info}")
        self.show_frame(img)
        self.show_info(info)

    def start(self):
        self.update_comboBox_default()
        try:
            if self.start_type == None:
                QMessageBox.information(self, "信息", "请先选择输入类型！", QMessageBox.Yes)
                return
            if self.start_type == 'img':
                # 读取中文路径下图片
                self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                _, result_info = self.predict_img(self.img)
                # 遍历字典
                for key, value in result_info.items():
                    # 判断 cls_name 值是否为 violence
                    if key == 'cls_name':
                        if value == 'violence':
                            # QMessageBox.information(self, "警告", "图片中存在暴力行为！", QMessageBox.Yes)
                            # 声音提醒
                            winsound.Beep(2500, 500)
                self.show_all(self.img_show, result_info)
            
            traceback.print_exc()

    def update_frame(self):
        if self.start_type == 'dir':
            # 循环遍历文件夹
            if not self.image_files:
                return
            if self.current_index >= len(self.image_files):
                self.current_index = 0  # 回到第一张，持续循环
            # 获取当前图像的名称和路径
            self.img_name = self.image_files[self.current_index]
            self.img_path = os.path.join(self.img_path_dir, self.img_name)
            # 读取图像并解码
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
            # 更新索引以处理下一张图像
            self.current_index += 1

        elif self.start_type in ['video', 'camera']:
            if self.start_type == 'video':
                # 读取下一帧
                ret, self.img = self.video.read()
                if not ret:
                    # 视频末尾，回到首帧继续
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, self.img = self.video.read()
                # 更新名称与路径
                frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                self.img_name = f"{self.video_name}_{frame_number}.jpg"
                self.img_path = self.video_path
            else:
                # 摄像头
                ret, self.img = self.video.read()
                if not ret:
                    # 读帧失败，尝试重连，不停止
                    try:
                        if self.video:
                            self.video.release()
                    except Exception:
                        pass
                    self.video = cv2.VideoCapture(self.video_name)
                    ret, self.img = self.video.read()
                self.frame_number += 1
                self.img_name = f"camera_{self.frame_number}.jpg"
                self.img_path = 'camera'

        # 确保图像存在
        if self.img is not None:
            # 进行图像预测
            results, result_info = self.predict_img(self.img)
            # 显示识别结果
            self.show_all(self.img_show, result_info)
        else:
            return
        # 对于视频，增加帧号以处理下一帧
        if self.start_type == 'video':
            self.frame_number += 1

    def predict_img(self, img):
        # 初始化结果信息字典
        result_info = {}
        # 记录开始时间以计算处理时间
        t1 = time.time()
        
        # 调试信息
        print(f"Debug: predict_img called with img shape: {img.shape if img is not None else 'None'}")
        print(f"Debug: self.img_name: {self.img_name}")
        print(f"Debug: self.result_img_path: {self.result_img_path}")
        
        # 不自动设置结果图像路径，等待保存时设置
        # 模型识别
        self.results = yolo.predict(img, imgsz=imgsz, conf=conf_thres, device=device, classes=classes)
        # 整理格式
        self.results = format_data(self.results)
        # 计算并记录消耗时间
        self.consum_time = str(round(time.time() - t1, 2)) + 's'
        # 记录输入时间
        self.input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 不自动写入文件，只显示识别信息表格
        self.show_table()
        # 增加编号
        self.number += 1
        # 获取下拉列表的值
        self.get_comboBox_value(self.results)

        if len(self.results) > 0:
            # 如果有识别结果，获取第一个结果的信息
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
        else:
            # 如果无识别结果，设置默认值
            box = [0, 0, 0, 0]
            score = 0
            cls_name = '无目标'

        # 格式化结果信息
        result_info = result_info_format(result_info, box, score, cls_name)
        
        # 在图像上绘制识别结果
        self.img_show = draw_info(img, self.results)
        
        # 不自动保存结果图像，等待用户点击保存按钮

        return self.results, result_info

    def get_advice(self):
        """获取防治建议"""
        try:
            # 检查是否有检测结果
            if not hasattr(self, 'results') or not self.results:
                self.textEdit_advice.setText("请先进行病虫害检测！")
                return

            # 检查当前模型配置
            if not current_model_config or not current_model_config.get('api_key'):
                current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)
                self.textEdit_advice.setText(
                    f"""❌ 未配置 {current_model_display} 模型的API Key！📝 当前配置：{current_model_display}""")
                return

            # 构建检测结果文本
            pest_names = []
            for result in self.results:
                if len(result) >= 3:
                    pest_name = result[0]
                    confidence = result[1]
                    chinese_pest_name = self.chinese_name.get(pest_name, pest_name)
                    pest_names.append(f"{chinese_pest_name}(置信度:{confidence:.2f})")

            pest_text = "、".join(pest_names)

            # 根据当前使用的AI模型动态显示加载提示
            current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)

            # 根据模型类型显示不同的加载提示
            if active_model == 'zhipu':
                loading_text = f"🚀 正在调用{curresplay}获取防治建议...\n\n⏳ 正在生成中，请稍候...\n📝 正在撰写科学严谨的防治技术报告...\n🔍 分析病虫害特征与发生规律...\n💡 制定综合防治技术方案...\n\n预计需要 20-35 秒，请耐心等待...\n💡 {current_model_display}为您提供1500字专业防治技术报告！"
          
            """.........."""



            self.textEdit_advice.setText(loading_text)
            if hasattr(self, 'pushButton_advice'):
                self.pushButton_advice.setEnabled(False)

            # 使用prompt_manager获取详细提示词（确保生成1500字报告）
            try:
                system_prompt = prompt_manager.get_prompt(active_model)
                prompt = f"{system_prompt}\n\n检测到的病虫害：{pest_text}\n\n请严格按照上述要求生成1500字的详细报告。"
            except Exception as e:
                print(f"加载提示词失败，使用默认提示词：{e}")
                prompt = f"请基于检测到的病虫害：{pest_text}，生成1500字的详细防治报告。"

            # 创建AI客户端
            ai_client = AIClient(current_model_config)

            # 在线程中请求（根据模型类型设置超时时间，适应1500字报告生成）
            if active_model == 'zhipu':
                actual_timeout = min(ai_timeout, 60)  # 智谱AI，最多60秒
            elif active_model == 'qianfan':
                actual_timeout = min(ai_timeout, 60)  # 百度千帆，最多60秒
            elif active_model == 'doubao':
                actual_timeout = min(ai_timeout, 60)  # 豆包大模型，最多60秒
            else:
                actual_timeout = min(ai_timeout, 60)  # 其他模型，最多60秒
            self.advice_thread = AdviceWorker(ai_client, prompt, actual_timeout)
            self.advice_thread.success.connect(self.on_advice_success)
            self.advice_thread.error.connect(self.on_advice_error)
            self.advice_thread.start()

            # 添加强制超时保护（根据模型类型设置，适应1500字报告生成）
            import threading
            def timeout_protection():
                import time
                if active_model == 'zhipu':
                    timeout_seconds = 70  # 智谱AI 70秒强制结束
                elif active_model == 'qianfan':
                    timeout_seconds = 70  # 百度千帆 70秒强制结束
                elif active_model == 'doubao':
                    timeout_seconds = 70  # 豆包大模型 70秒强制结束
                else:
                    timeout_seconds = 70  # 其他模型 70秒强制结束
                    
                time.sleep(timeout_seconds)
                if self.advice_thread and self.advice_thread.isRunning():
                    print(f"强制终止AI请求线程（{timeout_seconds}秒）")
                    self.advice_thread.terminate()
                    self.advice_thread.wait(1000)  # 等待1秒
                    self.on_advice_error(f"请求超时（{timeout_seconds}秒），已强制终止。\n请检查网络连接或稍后重试。")

            timeout_thread = threading.Thread(target=timeout_protection)
            timeout_thread.daemon = True
            timeout_thread.start()

        except Exception as e:
            self.textEdit_advice.setText(f"获取防治建议出错：\n{str(e)}")
            if hasattr(self, 'pushButton_advice'):
                self.pushButton_advice.setEnabled(True)

    def get_comboBox_value(self, results):
        '''
        获取当前所有的类别和ID，点击下拉列表时，使用
        '''
        # 默认第一个是 所有目标
        lst = ["所有目标"]
        for bbox in results:
            cls_name = bbox[0]
            lst.append(str(cls_name))
        self.comboBox_value = lst

    def show_info(self, result):
        try:

            if len(result) == 0:
                print("未识别到目标")
                return
            cls_name = result['cls_name']
            if len(self.chinese_name) > 3:
                cls_name = self.chinese_name[cls_name]
            if len(cls_name) > 10:
                # 当字符串太长时，显示不完整
                lst_cls_name = cls_name.split('_')
                cls_name = lst_cls_name[0][:10] + '...'

            self.label_class.setText(str(cls_name))
            self.label_score.setText(str(result['score']))
            self.label_xmin_v.setText(str(result['label_xmin_v']))
            self.label_ymin_v.setText(str(result['label_ymin_v']))
            self.label_xmax_v.setText(str(result['label_xmax_v']))
            self.label_ymax_v.setText(str(result['label_ymax_v']))
            self.update()  # 刷新界面
        except Exception as e:

            traceback.print_exc()

    def update_comboBox_default(self):
        """
        将下拉列表更新为 所有目标 默认状态
        """
        # 清空内容
        self.comboBox.clear()
        # 添加更新内容
        self.comboBox.addItems([self.comboBox_text])

    def show_table(self):
        try:
            # 显示表格
            self.RowLength = self.RowLength + 1
            self.tableWidget_info.setRowCount(self.RowLength)
            for column, content in enumerate(
                [self.number, self.img_path, self.input_time, self.results, len(self.results), self.consum_time,
                 "待保存"]):
                # self.tableWidget_info.setColumnWidth(3, 0)  # 将第二列的宽度设置为0，即不显示
                row = self.RowLength - 1
                item = QtWidgets.QTableWidgetItem(str(content))
                # 居中
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                # 设置字体颜色
                item.setForeground(QColor.fromRgb(column_color[0], column_color[1], column_color[2]))
                self.tableWidget_info.setItem(row, column, item)
            # 滚动到底部
            self.tableWidget_info.scrollToBottom()
        except Exception as e:
            traceback.print_exc()

    def write_files(self):
        """
        导出 excel、csv 数据
        """
        

    def save_current_result(self):
        """保存：每次点击在 output/类别名/时间戳/ 生成新的检测结果图与防治方案；无检测不保存"""
        try:
            # 无检测或无结果图则不保存
            if (self.img_show is None or self.img_name is None or not self.results):
                QMessageBox.information(self, "信息", "未检测到目标，\n未保存。", QMessageBox.Yes)
                return
            # 首次保存时创建输出目录与文件
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            
            # 仅保证输出根目录存在；不创建时间戳目录、result.txt、img_result
            self.result_session_dir = None
            self.result_txt = None
            self.result_img_path = None

            # 准备中文类别名（去重、保序）
            advice_txt = self.textEdit_advice.toPlainText().strip() if hasattr(self, 'textEdit_advice') else ''
            detected_names = []
            try:
                for r in (self.results or []):
                    name = r[0]
                    mapped = self.chinese_name.get(name, name) if hasattr(self, 'chinese_name') else name
                    detected_names.append(str(mapped))
            except Exception:
                pass
            if detected_names:
                seen = set()
                ordered = []
                for n in detected_names:
                    if n not in seen:
                        seen.add(n)
                        ordered.append(n)
                base_name = '+'.join(ordered)
            else:
                # 无检测则不保存
                QMessageBox.information(self, "信息", "未检测到目标，\n未保存。", QMessageBox.Yes)
                return
            for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
                base_name = base_name.replace(ch, '_')
            if len(base_name) > 60:
                base_name = base_name[:60]

            # 若存在同名目录，则追加(1)、(2)...
            base_dir_name = base_name
            category_dir = os.path.join(self.output_dir, base_dir_name)
            suffix = 1
            while os.path.exists(category_dir) and not os.listdir(category_dir):
                # 空目录可复用
                break
         
            # 不再写入 result.txt
                
            QMessageBox.information(self, "成功", f"已保存：\n{base_dir_name}\n检测结果图与防治方案。", QMessageBox.Yes)
        except Exception:
            traceback.print_exc()

    def on_advice_success(self, advice: str):
        """AI建议获取成功回调"""
        self.textEdit_advice.setText(advice)
        if hasattr(self, 'pushButton_advice'):
            self.pushButton_advice.setEnabled(True)

    def on_advice_error(self, msg: str):
        """AI建议获取失败回调"""
        current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)

        error_text = f"""❌ 获取防治建议失败

错误信息：{msg}

💡 可能的解决方案：
1. 检查网络连接是否正常
2. 确认API Key是否有效
3. 尝试重新点击"获取防治建议"按钮
4. 如果问题持续，请检查配置文件中的API设置

🔧 技术支持：
- 当前模型：{current_model_display}
- 超时设置：{ai_timeout}秒
- 最大Token：{ai_max_tokens}"""

        self.textEdit_advice.setText(error_text)
        if hasattr(self, 'pushButton_advice'):
            self.pushButton_advice.setEnabled(True)

    def show_model_info(self):
        """显示当前模型信息"""
        if not current_model_config:
            self.textEdit_advice.setText("❌ 未配置AI模型！")
            return

        current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)

        info = f"""当前AI模型配置

模型名称: {current_model_display}
API地址: {current_model_config.get('api_base', 'N/A')}
模型: {current_model_config.get('model', 'N/A')}
SSL验证: {current_model_config.get('verify_ssl', True)}
超时时间: {ai_timeout}秒
最大Token: {ai_max_tokens}
温度: {ai_temperature}

要切换模型，请修改 config/configs.yaml 中的 active_model 字段：
- 可选值: deepseek, qwen, openai, zhipu, qianfan, doubao, custom
- 重启程序即可生效"""
        self.textEdit_advice.setText(info)


# UI.ui转UI.py
# pyuic5 -x UI.ui -o UI.py
if __name__ == "__main__":
    path_cfg = 'config/configs.yaml'
    cfg = get_config()
    cfg.merge_from_file(path_cfg)
    # 加载模型相关的参数配置
    cfg_model = cfg.MODEL
    weights = cfg_model.WEIGHT
    conf_thres = float(cfg_model.CONF)
    classes = eval(cfg_model.CLASSES)
    imgsz = int(cfg_model.IMGSIZE)
    device = cfg_model.DEVICE
    # 加载UI界面相关的配置
    cfg_UI = cfg.UI
    background_img = cfg_UI.background
    padvalue = cfg_UI.padvalue
    column_widths = cfg_UI.column_widths
    column_color = cfg_UI.column_color
    title = cfg_UI.title
    label_title = cfg_UI.label_title
    zhutu2 = cfg_UI.zhutu2
    label_info_txt = cfg_UI.label_info_txt
    label_info_color = cfg_UI.label_info_color
    start_button_bg = cfg_UI.start_button_bg
    start_button_font = cfg_UI.start_button_font
    export_button_bg = cfg_UI.export_button_bg
    export_button_font = cfg_UI.export_button_font
    label_control_color = cfg_UI.label_control_color
    label_img_color = cfg_UI.label_img_color
    header_background_color = cfg_UI.table_widget_info_styles.header_background_color
    header_color = cfg_UI.tabl
    item_hover_background_color = cfg_UI.table_widget_info_styles.item_hover_background_color
    # 加载通用配置
    camera_num = int(cfg.CONFIG.camera_num)
    chinese_name = cfg.CONFIG.chinese_name

    # 模型加载
    yolo = YOLO(weights)
    # 模型预热
    yolo.predict(np.zeros((300, 300, 3), dtype='uint8'), device=device)

    # 创建QApplication实例
    app = QApplication([])
    # 创建自定义的主窗口对象
    window = MyMainWindow(cfg)
    # 显示窗口
    window.show()
    # 运行应用程序
    app.exec_()
