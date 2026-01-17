"""
OCR 引擎模块
支持多种 OCR 引擎: PaddleOCR, macOS 原生 OCR
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np


class OCRResult:
    """OCR 识别结果的统一数据结构"""

    def __init__(self, texts: List[Dict], polys: List[np.ndarray] = None):
        """
        Args:
            texts: 文字信息列表, 每项包含 {text, confidence, box}
            polys: 多边形坐标列表 (numpy 数组格式)
        """
        self.texts = texts
        self.polys = polys if polys is not None else []

    def __len__(self):
        return len(self.texts)

    def __iter__(self):
        return iter(self.texts)


class OCREngine(ABC):
    """OCR 引擎抽象基类"""

    @abstractmethod
    def recognize(self, image_path: str) -> OCRResult:
        """
        识别图片中的文字

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 对象
        """
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """返回引擎名称"""
        pass


class PaddleOCREngine(OCREngine):
    """PaddleOCR 引擎"""

    def __init__(self, use_angle_cls: bool = True, lang: str = "ch"):
        """
        初始化 PaddleOCR

        Args:
            use_angle_cls: 是否使用角度分类
            lang: 语言 (ch=中文, en=英文)
        """
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
            print(f"✓ PaddleOCR 引擎初始化成功 (语言: {lang})")
        except ImportError:
            raise ImportError("请先安装 PaddleOCR: pip install paddleocr")

    def recognize(self, image_path: str) -> OCRResult:
        """
        使用 PaddleOCR 识别图片

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 对象
        """
        result = self.ocr.ocr(str(image_path))

        texts = []
        polys_list = []

        if not result:
            return OCRResult(texts, polys_list)

        for page_result in result:
            # 新版 PaddleOCR 返回的是 OCRResult 对象(可以像字典一样访问)
            if isinstance(page_result, dict) and 'rec_texts' in page_result:
                rec_texts = page_result['rec_texts']
                rec_scores = page_result.get('rec_scores', [])
                dt_polys = page_result.get('dt_polys', [])

                for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    box = dt_polys[i] if i < len(dt_polys) else None
                    texts.append({
                        "text": text,
                        "confidence": round(float(score), 4),
                        "box": str(box) if box is not None else None
                    })
                    # 保存numpy数组格式的坐标用于绘制
                    if box is not None and isinstance(box, np.ndarray):
                        polys_list.append(box)

            # 兼容旧版列表格式
            elif isinstance(page_result, (list, tuple)) and page_result:
                for line in page_result:
                    if line and len(line) >= 2:
                        if isinstance(line[1], (tuple, list)) and len(line[1]) >= 2:
                            text = line[1][0]
                            confidence = line[1][1]
                            texts.append({
                                "text": text,
                                "confidence": round(confidence, 4),
                                "box": line[0]
                            })
                            if isinstance(line[0], np.ndarray):
                                polys_list.append(line[0])

        return OCRResult(texts, polys_list)

    def get_engine_name(self) -> str:
        return "PaddleOCR"


class MacOSNativeOCREngine(OCREngine):
    """macOS 原生 OCR 引擎 (使用 Vision Framework)"""

    def __init__(self, lang: str = "zh-Hans"):
        """
        初始化 macOS 原生 OCR

        Args:
            lang: 语言代码 (zh-Hans=简体中文, zh-Hant=繁体中文, en-US=英文)
        """
        import platform
        if platform.system() != "Darwin":
            raise OSError("macOS 原生 OCR 仅支持在 macOS 系统上运行")

        try:
            import Quartz
            import Vision
            import objc
            self.Vision = Vision
            self.Quartz = Quartz
            self.lang = lang
            print(f"✓ macOS 原生 OCR 引擎初始化成功 (语言: {lang})")
        except ImportError:
            raise ImportError(
                "请先安装依赖: pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
            )

    def recognize(self, image_path: str) -> OCRResult:
        """
        使用 macOS Vision Framework 识别图片

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 对象
        """
        from Foundation import NSURL, NSArray

        # 加载图片
        image_url = NSURL.fileURLWithPath_(str(image_path))

        # 创建识别请求
        request = self.Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(self.Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([self.lang])
        request.setUsesLanguageCorrection_(True)

        # 创建请求处理器
        requests = NSArray.arrayWithObject_(request)

        # 加载图片并处理
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # 使用 CGImageSource 创建图片
        from Foundation import NSData
        ns_data = NSData.dataWithBytes_length_(image_data, len(image_data))
        image_source = self.Quartz.CGImageSourceCreateWithData(ns_data, None)

        if not image_source:
            return OCRResult([], [])

        cg_image = self.Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if not cg_image:
            return OCRResult([], [])

        # 创建请求处理器并执行
        handler = self.Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}
        )

        error = None
        success = handler.performRequests_error_(requests, error)

        if not success or error:
            return OCRResult([], [])

        # 提取识别结果
        texts = []
        polys_list = []

        observations = request.results()
        if observations:
            # 获取图片尺寸
            width = self.Quartz.CGImageGetWidth(cg_image)
            height = self.Quartz.CGImageGetHeight(cg_image)

            for observation in observations:
                text = observation.text()
                confidence = observation.confidence()

                # 获取边界框 (Vision 坐标系是左下角为原点)
                bounding_box = observation.boundingBox()

                # 转换坐标到 OpenCV 格式 (左上角为原点)
                x = bounding_box.origin.x * width
                y = (1 - bounding_box.origin.y - bounding_box.size.height) * height
                w = bounding_box.size.width * width
                h = bounding_box.size.height * height

                # 创建矩形坐标 (四个角点)
                poly = np.array([
                    [x, y],  # 左上
                    [x + w, y],  # 右上
                    [x + w, y + h],  # 右下
                    [x, y + h]  # 左下
                ], dtype=np.float32)

                texts.append({
                    "text": str(text),
                    "confidence": round(float(confidence), 4),
                    "box": str(poly)
                })
                polys_list.append(poly)

        return OCRResult(texts, polys_list)

    def get_engine_name(self) -> str:
        return "macOS Native OCR"


class OCRMacEngine(OCREngine):
    """ocrmac 引擎 (第三方命令行工具)"""

    def __init__(self):
        """
        初始化 ocrmac 引擎
        需要先安装: brew install ocrmac
        """
        import platform
        import subprocess

        if platform.system() != "Darwin":
            raise OSError("ocrmac 仅支持在 macOS 系统上运行")

        # 检查 ocrmac 是否安装
        try:
            result = subprocess.run(
                ["ocrmac", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError("ocrmac 未安装或无法运行")
            print("✓ ocrmac 引擎初始化成功")
        except FileNotFoundError:
            raise FileNotFoundError(
                "请先安装 ocrmac: brew install ocrmac"
            )

    def recognize(self, image_path: str) -> OCRResult:
        """
        使用 ocrmac 识别图片

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 对象
        """
        import subprocess
        import cv2

        # 使用 ocrmac 进行识别
        try:
            result = subprocess.run(
                ["ocrmac", str(image_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return OCRResult([], [])

            # ocrmac 只返回文本,不返回坐标和置信度
            text_output = result.stdout.strip()

            if not text_output:
                return OCRResult([], [])

            # 将整个文本作为一个结果
            # 由于 ocrmac 不提供坐标,我们使用整个图片的边界
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                poly = np.array([
                    [0, 0],
                    [w, 0],
                    [w, h],
                    [0, h]
                ], dtype=np.float32)

                texts = [{
                    "text": text_output,
                    "confidence": 1.0,  # ocrmac 不提供置信度
                    "box": str(poly)
                }]
                polys_list = [poly]

                return OCRResult(texts, polys_list)

            return OCRResult([{
                "text": text_output,
                "confidence": 1.0,
                "box": None
            }], [])

        except subprocess.TimeoutExpired:
            print("  ✗ ocrmac 执行超时")
            return OCRResult([], [])
        except Exception as e:
            print(f"  ✗ ocrmac 执行错误: {e}")
            return OCRResult([], [])

    def get_engine_name(self) -> str:
        return "ocrmac"


def create_ocr_engine(engine_type: str = "paddle", **kwargs) -> OCREngine:
    """
    工厂函数: 创建 OCR 引擎

    Args:
        engine_type: 引擎类型 ("paddle", "macos", "ocrmac")
        **kwargs: 传递给引擎构造函数的参数

    Returns:
        OCREngine 实例

    Examples:
        >>> # 使用 PaddleOCR
        >>> engine = create_ocr_engine("paddle", lang="ch")
        >>>
        >>> # 使用 macOS 原生 OCR
        >>> engine = create_ocr_engine("macos", lang="zh-Hans")
        >>>
        >>> # 使用 ocrmac
        >>> engine = create_ocr_engine("ocrmac")
    """
    engine_type = engine_type.lower()

    if engine_type == "paddle":
        return PaddleOCREngine(**kwargs)
    elif engine_type == "macos":
        return MacOSNativeOCREngine(**kwargs)
    elif engine_type == "ocrmac":
        return OCRMacEngine(**kwargs)
    else:
        raise ValueError(
            f"不支持的引擎类型: {engine_type}. "
            f"支持的类型: paddle, macos, ocrmac"
        )
