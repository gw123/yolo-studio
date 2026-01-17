import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ocr_engine import create_ocr_engine, OCREngine, OCRResult


def get_chinese_font(font_size: int = 20):
    """
    获取中文字体

    Args:
        font_size: 字体大小

    Returns:
        PIL ImageFont 对象
    """
    # macOS 系统中文字体路径
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方字体
        "/System/Library/Fonts/STHeiti Light.ttc",  # macOS 华文黑体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Windows\\Fonts\\simhei.ttf",  # Windows 黑体
        "C:\\Windows\\Fonts\\msyh.ttf",  # Windows 微软雅黑
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue

    # 如果找不到字体,使用默认字体
    return ImageFont.load_default()


def cv2_add_chinese_text(img: np.ndarray, text: str, position: tuple,
                         font_size: int = 20, color: tuple = (0, 255, 0)):
    """
    在 OpenCV 图片上添加中文文字

    Args:
        img: OpenCV 图片 (BGR 格式)
        text: 要添加的文字
        position: 文字位置 (x, y)
        font_size: 字体大小
        color: 文字颜色 (BGR 格式)

    Returns:
        添加文字后的图片
    """
    # 转换为 PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 获取字体
    font = get_chinese_font(font_size)

    # 转换颜色为 RGB
    color_rgb = (color[2], color[1], color[0])

    # 绘制文字
    draw.text(position, text, font=font, fill=color_rgb)

    # 转换回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_image_files(data_dir: str, extensions: set = None) -> List[Path]:
    """
    获取目录下所有图片文件

    Args:
        data_dir: 图片所在目录
        extensions: 支持的图片格式扩展名集合

    Returns:
        图片文件路径列表
    """
    if extensions is None:
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"目录 {data_dir} 不存在")

    image_files = [
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    return image_files


def extract_ocr_results(result: OCRResult) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    从 OCR 结果中提取文字信息和多边形坐标

    Args:
        result: OCRResult 对象

    Returns:
        (文字信息列表, 多边形坐标列表)
    """
    return result.texts, result.polys


def draw_boxes_with_numbers(img: np.ndarray, polys_list: List[np.ndarray]) -> np.ndarray:
    """
    在图片上绘制带序号的检测框

    Args:
        img: 原始图片
        polys_list: 多边形坐标列表

    Returns:
        标注后的图片
    """
    img_copy = img.copy()

    for i, poly in enumerate(polys_list):
        if isinstance(poly, np.ndarray):
            # 转换为整数坐标
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            # 绘制绿色多边形框,线宽2
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)

            # 在框的左上角添加序号
            if len(poly) > 0:
                top_left = tuple(poly[0].astype(np.int32))
                # 绘制绿色圆形背景
                cv2.circle(img_copy, top_left, 12, (0, 255, 0), -1)
                # 绘制黑色序号
                cv2.putText(img_copy, str(i+1),
                           (top_left[0]-6, top_left[1]+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img_copy


def draw_boxes_with_text(img: np.ndarray, polys_list: List[np.ndarray],
                         texts: List[Dict]) -> np.ndarray:
    """
    在图片上绘制带文字标注的检测框 (支持中文)

    Args:
        img: 原始图片
        polys_list: 多边形坐标列表
        texts: 文字信息列表

    Returns:
        标注后的图片
    """
    img_copy = img.copy()

    # 先绘制所有的框
    for poly in polys_list:
        if isinstance(poly, np.ndarray):
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)

    # 然后使用 PIL 添加中文文字
    for i, (poly, text_info) in enumerate(zip(polys_list, texts)):
        if isinstance(poly, np.ndarray) and len(poly) > 0:
            top_left = poly[0].astype(np.int32)
            text = text_info['text'][:20]  # 限制文字长度

            # 计算文字位置(框的上方)
            x, y = int(top_left[0]), int(top_left[1]) - 5

            # 使用 PIL 添加中文文字
            img_copy = cv2_add_chinese_text(
                img_copy,
                text,
                (x, max(0, y - 20)),  # 确保不超出图片边界
                font_size=16,
                color=(0, 255, 0)
            )

    return img_copy


def save_annotated_images(image_file: Path, polys_list: List[np.ndarray],
                         texts: List[Dict], output_path: Path) -> None:
    """
    保存标注后的图片(包含两个版本)

    Args:
        image_file: 原始图片路径
        polys_list: 多边形坐标列表
        texts: 文字信息列表
        output_path: 输出目录路径
    """
    img = cv2.imread(str(image_file))
    if img is None or not polys_list:
        return

    # 保存带序号的版本
    img_with_numbers = draw_boxes_with_numbers(img, polys_list)
    output_image_path = output_path / f"annotated_{image_file.name}"
    cv2.imwrite(str(output_image_path), img_with_numbers)
    print(f"  ✓ 标注图片已保存: {output_image_path}")

    # 保存带文字的版本
    img_with_text = draw_boxes_with_text(img, polys_list, texts)
    output_text_path = output_path / f"annotated_with_text_{image_file.name}"
    cv2.imwrite(str(output_text_path), img_with_text)
    print(f"  ✓ 文字标注图片已保存: {output_text_path}")


def print_ocr_result(texts: List[Dict], elapsed_time: float) -> None:
    """
    打印 OCR 识别结果

    Args:
        texts: 文字信息列表
        elapsed_time: 处理耗时
    """
    print(f"  ✓ 完成 - 耗时: {elapsed_time:.3f}秒")
    print(f"  ✓ 识别到 {len(texts)} 条文字")

    if texts:
        print("  识别内容:")
        for text_info in texts[:5]:  # 只显示前5条
            print(f"    - {text_info['text']} (置信度: {text_info['confidence']:.2%})")
        if len(texts) > 5:
            print(f"    ... 还有 {len(texts) - 5} 条")
    else:
        print("  (未识别到文字内容)")


def process_single_image(ocr: OCREngine, image_file: Path,
                        output_path: Path) -> Dict:
    """
    处理单张图片

    Args:
        ocr: OCREngine 实例
        image_file: 图片文件路径
        output_path: 输出目录路径

    Returns:
        处理结果字典
    """
    start_time = time.time()

    try:
        # 执行 OCR
        result = ocr.recognize(str(image_file))

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 提取文字和坐标
        texts, polys_list = extract_ocr_results(result)

        # 构建结果
        image_result = {
            "image_name": image_file.name,
            "image_path": str(image_file),
            "processing_time": round(elapsed_time, 3),
            "text_count": len(texts),
            "ocr_result": texts
        }

        # 打印结果
        print_ocr_result(texts, elapsed_time)

        # 保存标注图片
        if texts:
            save_annotated_images(image_file, polys_list, texts, output_path)

        return image_result

    except Exception as e:
        import traceback
        print(f"  ✗ 错误: {str(e)}")
        print(f"  详细错误信息:")
        traceback.print_exc()

        return {
            "image_name": image_file.name,
            "image_path": str(image_file),
            "processing_time": 0,
            "error": str(e)
        }


def save_results_to_json(results: Dict, output_file: str) -> None:
    """
    保存结果到 JSON 文件

    Args:
        results: 结果字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(image_files: List[Path], total_elapsed: float,
                 output_file: str, output_dir: str) -> None:
    """
    打印处理总结

    Args:
        image_files: 图片文件列表
        total_elapsed: 总耗时
        output_file: JSON 输出文件路径
        output_dir: 图片输出目录
    """
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"总耗时: {total_elapsed:.3f}秒")
    print(f"平均耗时: {total_elapsed/len(image_files):.3f}秒/图片")
    print(f"结果已保存到: {output_file}")
    print(f"标注图片已保存到: {output_dir}/ 目录")
    print("=" * 60)


def process_images(data_dir="./data", output_file="ocr_results.json",
                  output_dir="./output", engine="paddle", **engine_kwargs):
    """
    批量处理图片进行 OCR 识别

    Args:
        data_dir: 图片所在目录
        output_file: 输出的 JSON 文件路径
        output_dir: 标注图片的输出目录
        engine: OCR 引擎类型 ("paddle", "macos", "ocrmac")
        **engine_kwargs: 传递给 OCR 引擎的参数
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 初始化 OCR 引擎
    print(f"初始化 OCR 引擎 ({engine})...")
    try:
        ocr = create_ocr_engine(engine, **engine_kwargs)
    except Exception as e:
        print(f"✗ OCR 引擎初始化失败: {e}")
        return

    # 获取所有图片文件
    try:
        image_files = get_image_files(data_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    if not image_files:
        print(f"警告: 在 {data_dir} 目录下没有找到图片文件")
        return

    print(f"\n找到 {len(image_files)} 个图片文件")
    print("=" * 60)

    # 存储所有结果
    all_results = {
        "total_images": len(image_files),
        "total_time": 0,
        "results": []
    }

    total_start_time = time.time()

    # 处理每个图片
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_file.name}")

        image_result = process_single_image(ocr, image_file, output_path)
        all_results["results"].append(image_result)

    # 计算总耗时
    total_elapsed = time.time() - total_start_time
    all_results["total_time"] = round(total_elapsed, 3)

    # 保存结果到 JSON 文件
    save_results_to_json(all_results, output_file)

    # 打印总结
    print_summary(image_files, total_elapsed, output_file, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量 OCR 图片识别工具")
    parser.add_argument("--data-dir", default="./data", help="图片所在目录")
    parser.add_argument("--output-file", default="ocr_results.json", help="JSON 输出文件")
    parser.add_argument("--output-dir", default="./output", help="标注图片输出目录")
    parser.add_argument("--engine", default="paddle",
                       choices=["paddle", "macos", "ocrmac"],
                       help="OCR 引擎类型")
    parser.add_argument("--lang", default="ch",
                       help="语言设置 (paddle: ch/en, macos: zh-Hans/zh-Hant/en-US)")

    args = parser.parse_args()

    # 根据引擎类型设置参数
    engine_kwargs = {}
    if args.engine == "paddle":
        engine_kwargs = {"use_angle_cls": True, "lang": args.lang}
    elif args.engine == "macos":
        # 转换语言代码
        lang_map = {"ch": "zh-Hans", "en": "en-US"}
        engine_kwargs = {"lang": lang_map.get(args.lang, args.lang)}

    process_images(
        data_dir=args.data_dir,
        output_file=args.output_file,
        output_dir=args.output_dir,
        engine=args.engine,
        **engine_kwargs
    )
