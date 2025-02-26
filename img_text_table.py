import os
import json
import fitz            # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import functools
'''
crop ->  이미지 텍스트 표 분리 -> sort_text 

본 파일에서는 crop.py에서 추출한 바운딩 박스 정보를 활용하여
PDF 페이지를 구조화하는 예시 코드를 제공. (아직 sort 하지 않음.)

주요 로직:
1) 바운딩 박스 정보를 PDF 페이지 좌표로 변환
2) 텍스트, 이미지, 표 블록을 처리하는 핸들러 함수 정의
3) 페이지별로 바운딩 박스 정보를 활용하여 PDF 페이지 구조화
4) 최종 결과를 JSON으로 저장

'''
#########################
# 1) 핸들러 함수들
#########################

def handle_text_block(pdf_page_plumb, bbox_plumber):
    """
    텍스트 블록을 처리하는 함수.
    pdfplumber의 크롭 영역(bbox_plumber)에서 텍스트를 추출하고,
    줄 단위로 청크화.
    """
    cropped = pdf_page_plumb.within_bbox(bbox_plumber)
    extracted_text = cropped.extract_text() or ""
    lines = extracted_text.splitlines()
    chunk_list = [line.strip() for line in lines if line.strip()]
    return {
        "raw_text": extracted_text,
        "chunks": chunk_list
    }

def handle_figure_block(pil_img, x1_img, y1_img, x2_img, y2_img, output_dir, page_num, class_id, conf):
    """
    이미지(figure) 블록을 처리하는 함수.
    Pillow로 원본 이미지를 Crop하여 파일로 저장하고 경로 반환.
    """
    left = min(x1_img, x2_img)
    right = max(x1_img, x2_img)
    upper = min(y1_img, y2_img)
    lower = max(y1_img, y2_img)
    cropped_img = pil_img.crop((left, upper, right, lower))
    fig_filename = f"page_{page_num}_figure_{class_id}_{int(conf*100)}.jpg"
    fig_path = os.path.join(output_dir, fig_filename)
    cropped_img.save(fig_path)
    return {
        "image_file": fig_path
    }

def handle_table_block():
    """
    표(table) 블록을 처리하는 함수.
    현재는 간단히 '추가 처리 없음' 정보를 반환합니다.
    """
    return {
        "info": "table block - no extra parsing yet"
    }

#########################
# 2) PDF 및 이미지 로드 및 설정
#########################
pdf_path = r"C:\Users\xpc\Desktop\sample.pdf"
json_path = r"C:\Users\xpc\Desktop\annotated_images\sample_bounding_boxes.json"
output_dir = r"C:\Users\xpc\Desktop\output"
os.makedirs(output_dir, exist_ok=True)
dpi = 300

'''
{0: 'title', 1: 'plain text', 2: 'abandoned', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
텍스트  : 0,1,4,6,7,9
이미지 : 3 + 4(figure_caption)
표 : 5 + 6(table_caption) + 7(table_footnote)


'''

# 클래스 분류 (예시)
TEXT_CLASSES   = [0, 1, 4, 6, 7, 9]
FIGURE_CLASSES = [3, 4]
TABLE_CLASSES  = [5, 6, 7]

# (옵션) bbox 보정 (회전/반전)
FORCE_180_ROTATION = True
FORCE_HORIZONTAL_FLIP = True

def rotate_bbox_180(x1, y1, x2, y2, pdf_width, pdf_height):
    """
    bounding box를 180도 회전(또는 되돌리기)하는 함수.
    (x1, y1), (x2, y2)는 PDF 좌표(왼쪽 하단 원점)에서 이미 정렬된 상태라고 가정.
    (x, y) → (pdf_width - x, pdf_height - y)
    반환 값은 (min_x, min_y, max_x, max_y) 형태입니다.
    """
    new_x1 = pdf_width - x2
    new_x2 = pdf_width - x1
    new_y1 = pdf_height - y2
    new_y2 = pdf_height - y1
    new_x1, new_x2 = sorted([new_x1, new_x2])
    new_y1, new_y2 = sorted([new_y1, new_y2])
    return (new_x1, new_y1, new_x2, new_y2)

def flip_bbox_horizontal(x1, y1, x2, y2, pdf_width, pdf_height):
    """
    좌/우 반전(Left-Right flip) 함수.
    (x1, y1, x2, y2)는 PDF 좌표(왼쪽 하단 원점)에서 이미 정렬된 상태라고 가정.
    (x, y) → (pdf_width - x, y)
    """
    new_x1 = pdf_width - x2
    new_x2 = pdf_width - x1
    new_y1 = y1
    new_y2 = y2
    new_x1, new_x2 = sorted([new_x1, new_x2])
    return (new_x1, new_y1, new_x2, new_y2)

#########################
# 3) PDF 및 JSON 로드, 블록 추출
#########################
print("[INFO] pdfplumber, PyMuPDF, pdf2image 로딩...")
doc = fitz.open(pdf_path)               # PyMuPDF
pdf_plumb = pdfplumber.open(pdf_path)     # pdfplumber
images = convert_from_path(pdf_path, dpi=dpi)
print(f"[INFO] 총 {len(doc)}페이지 / 이미지 {len(images)}개 로드 완료.")

with open(json_path, "r", encoding="utf-8") as f:
    yolo_results = json.load(f)

all_pages_data = {}

#########################
# 4) 페이지별 처리 (정렬 함수 제거, 변환된 순서 그대로 사용)
#########################
for page_key, blocks in yolo_results.items():
    try:
        page_num = int(page_key.split('_')[1])
    except Exception as e:
        print(f"[WARN] 페이지 키 파싱 오류: {page_key} - {e}")
        continue

    if page_num > len(doc) or page_num > len(images):
        print(f"[WARN] 페이지 {page_num}이(가) PDF 또는 이미지에 없음!")
        continue

    pdf_page_fitz = doc[page_num - 1]
    pdf_page_plumb = pdf_plumb.pages[page_num - 1]
    pil_img = images[page_num - 1]

    pdf_width = pdf_page_fitz.rect.width
    pdf_height = pdf_page_fitz.rect.height
    img_width, img_height = pil_img.size

    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    converted_blocks = []
    for block in blocks:
        x1_img, y1_img, x2_img, y2_img = block["x1"], block["y1"], block["x2"], block["y2"]
        class_id = block["class_id"]
        conf = block["confidence"]

        # 이미지 좌표 → PDF 좌표 변환
        pdf_x1 = x1_img / scale_x
        pdf_x2 = x2_img / scale_x
        pdf_y1 = pdf_height - (y1_img / scale_y)
        pdf_y2 = pdf_height - (y2_img / scale_y)

        pdf_y_bottom = min(pdf_y1, pdf_y2)
        pdf_y_top = max(pdf_y1, pdf_y2)

        # (옵션) 회전/반전
        if FORCE_180_ROTATION:
            pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top = rotate_bbox_180(
                pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top,
                pdf_width, pdf_height
            )
        if FORCE_HORIZONTAL_FLIP:
            pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top = flip_bbox_horizontal(
                pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top,
                pdf_width, pdf_height
            )

        # margin (텍스트 마지막줄 잘림 방지)
        margin = 2.0
        pdf_x1 = max(pdf_x1 - margin, 0)
        pdf_y_bottom = max(pdf_y_bottom - margin, 0)
        pdf_x2 = min(pdf_x2 + margin, pdf_width)
        pdf_y_top = min(pdf_y_top + margin, pdf_height)

        converted_blocks.append({
            "class_id": class_id,
            "confidence": conf,
            "pdf_bbox": (pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top),
            "x1_img": x1_img, "y1_img": y1_img, "x2_img": x2_img, "y2_img": y2_img
        })

    # 정렬 x -> 이후 파일에서 정렬해서 사용할 것.
    blocks_to_process = converted_blocks

    page_data = {
        "text_blocks": [],
        "figure_blocks": [],
        "table_blocks": []
    }

    for sb in blocks_to_process:
        cid = sb["class_id"]
        conf = sb["confidence"]
        px1, pyb, px2, pyt = sb["pdf_bbox"]

        # pdfplumber에 전달할 bbox는 (min_x, min_y, max_x, max_y)
        bbox = (min(px1, px2), min(pyb, pyt), max(px1, px2), max(pyb, pyt))

        if cid in TEXT_CLASSES:
            cropped = pdf_page_plumb.within_bbox(bbox)
            extracted_text = cropped.extract_text() or ""
            lines = extracted_text.splitlines()
            chunk_list = [line.strip() for line in lines if line.strip()]
            page_data["text_blocks"].append({
                "class_id": cid,
                "confidence": conf,
                "pdf_bbox": bbox,
                "raw_text": extracted_text,
                "chunks": chunk_list
            })

        elif cid in FIGURE_CLASSES:
            left = min(sb["x1_img"], sb["x2_img"])
            right = max(sb["x1_img"], sb["x2_img"])
            upper = min(sb["y1_img"], sb["y2_img"])
            lower = max(sb["y1_img"], sb["y2_img"])
            cropped_img = pil_img.crop((left, upper, right, lower))
            fig_filename = f"page_{page_num}_figure_{cid}_{int(conf*100)}.jpg"
            fig_path = os.path.join(output_dir, fig_filename)
            cropped_img.save(fig_path)
            page_data["figure_blocks"].append({
                "class_id": cid,
                "confidence": conf,
                "pdf_bbox": bbox,
                "image_file": fig_path
            })

        elif cid in TABLE_CLASSES:
            page_data["table_blocks"].append({
                "class_id": cid,
                "confidence": conf,
                "pdf_bbox": bbox,
                "info": "table block - no extra parsing yet"
            })

    all_pages_data[f"page_{page_num}"] = page_data

#########################
# 5) 결과 JSON 저장
#########################
final_json_path = os.path.join(output_dir, "structured_data.json")
with open(final_json_path, "w", encoding="utf-8") as f:
    json.dump(all_pages_data, f, indent=4, ensure_ascii=False)

doc.close()
pdf_plumb.close()
print("[DONE] 텍스트/이미지/표 데이터 구조화 완료.")
print(f"결과: {final_json_path}")
