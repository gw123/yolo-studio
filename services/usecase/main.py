import pyautogui
import pyperclip
import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from ocrmac import ocrmac
from paddleocr import PaddleOCR


# ================= 1. OCR 适配器模式设计 =================

class OCRAdapter(ABC):
    """OCR 引擎抽象基类"""

    @abstractmethod
    def recognize(self, image_path):
        """返回格式: [{'text': '内容', 'box': [x, y, w, h], 'confidence': 0.9}, ...]"""
        pass


class PaddleOCRAdapter(OCRAdapter):
    """百度 PaddleOCR 适配器"""

    def __init__(self):
        # 首次初始化会下载模型
        self.engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

    def recognize(self, image_path):
        results = self.engine.ocr(image_path, cls=True)
        final_results = []
        if results[0] is None: return []
        for line in results[0]:
            box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            conf = line[1][1]
            # 转换为 [x, y, w, h]
            x, y = box[0]
            w = box[1][0] - box[0][0]
            h = box[2][1] - box[1][1]
            final_results.append({'text': text, 'box': [x, y, w, h], 'confidence': conf})
        return final_results


class MacOCRAdapter(OCRAdapter):
    """macOS 原生 OCR (ocrmac) 适配器"""

    def recognize(self, image_path):
        # annotations 格式: (text, confidence, [x, y, w, h])
        annotations = ocrmac.OCR(image_path).recognize()
        final_results = []
        for text, conf, bbox in annotations:
            final_results.append({'text': text, 'box': bbox, 'confidence': conf})
        return final_results


# ================= 2. 核心自动化工具类 =================

class AutoRobot:
    def __init__(self, ocr_engine: OCRAdapter):
        self.ocr = ocr_engine
        # macOS Retina 屏幕缩放因子，通常为 2
        # PyAutoGUI 截图是物理像素，但点击需要逻辑像素
        self.scale = 2 if pyautogui.size()[0] < 2000 else 1

    def input_chinese(self, text):
        """使用剪贴板输入中文"""
        pyperclip.copy(text)
        time.sleep(0.2)
        pyautogui.hotkey('command', 'v')

    def find_and_click_text(self, target_text, screenshot_name="temp_screen.png"):
        """识别屏幕文字并点击"""
        # 1. 截图
        pyautogui.screenshot(screenshot_name)

        # 2. OCR 识别
        print(f"正在使用 {self.ocr.__class__.__name__} 识别...")
        results = self.ocr.recognize(screenshot_name)

        for item in results:
            if target_text in item['text']:
                x, y, w, h = item['box']
                # 计算中心点并适配 Retina 缩放
                center_x = (x + w / 2) / self.scale
                center_y = (y + h / 2) / self.scale

                print(f"找到文本 '{item['text']}'，坐标: ({center_x}, {center_y})")
                pyautogui.click(center_x, center_y)
                return True
        print(f"未找到目标文字: {target_text}")
        return False

    def find_image_cv2(self, template_path, threshold=0.8):
        """使用 OpenCV 进行模板匹配 (找图)"""
        screen = pyautogui.screenshot()
        screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        template = cv2.imread(template_path)

        res = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            h, w = template.shape[:2]
            center_x = (max_loc[0] + w // 2) / self.scale
            center_y = (max_loc[1] + h // 2) / self.scale
            return center_x, center_y
        return None


# ================= 3. 实际业务逻辑测试 =================

if __name__ == "__main__":
    # --- 切换引擎只需改这里 ---
    # 方式 A: 使用百度 PaddleOCR (更准)
    # engine = PaddleOCRAdapter()

    # 方式 B: 使用 Mac 原生 OCR (更快)
    engine = MacOCRAdapter()

    bot = AutoRobot(engine)

    print("脚本启动，请切换到目标窗口...")
    time.sleep(3)

    # 1. 测试 OCR 查找并点击
    if bot.find_and_click_text("搜索"):
        time.sleep(1)
        # 2. 测试中文输入
        bot.input_chinese("如何学习Python自动化")
        pyautogui.press('enter')

    # 3. 测试 OpenCV 找图点击 (假设你有一个 icon.png)
    # pos = bot.find_image_cv2('icon.png')
    # if pos:
    #     pyautogui.click(pos)