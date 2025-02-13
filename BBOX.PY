import fitz  # PyMuPDF
import json
from pdf2image import convert_from_path

# 파일 경로 설정
pdf_path = r"C:\Users\xpc\Desktop\CLIP.pdf"
json_path = r"C:\Users\xpc\Desktop\annotated_images\bounding_boxes.json"
output_pdf_path = r"C:\Users\xpc\Desktop\output\CLIP_annotated.pdf"

# JSON 파일 읽기 (YOLO 결과)
with open(json_path, "r", encoding="utf-8") as f:
    yolo_results = json.load(f)

# PDF 파일 열기 (PyMuPDF)
doc = fitz.open(pdf_path)

# pdf2image로 PDF → 이미지 변환 (300dpi)
images = convert_from_path(pdf_path, dpi=300)

def rotate_bbox_180(x1, y1, x2, y2, pdf_width, pdf_height):
    """
    bounding box를 180도 회전(또는 되돌리기)하는 함수.
    (x1, y1), (x2, y2)는 PDF 좌표(왼쪽 하단 원점)에서 이미 정렬된 상태라고 가정.
    (x, y) → (pdf_width - x, pdf_height - y)
    """
    new_x1 = pdf_width - x2
    new_x2 = pdf_width - x1
    new_y1 = pdf_height - y2
    new_y2 = pdf_height - y1

    # x1 < x2, y1 < y2 정렬
    new_x1, new_x2 = sorted([new_x1, new_x2])
    new_y1, new_y2 = sorted([new_y1, new_y2])

    return (new_x1, new_y1, new_x2, new_y2)

def flip_bbox_horizontal(x1, y1, x2, y2, pdf_width, pdf_height):
    """
    좌/우 반전(Left-Right flip) 함수.
    (x1, y1, x2, y2)는 PDF 좌표(왼쪽 하단 원점)에서 이미 정렬된 상태라고 가정.
    (x, y) → (pdf_width - x, y)
    즉, x 좌표만 뒤집고, y 좌표는 그대로 둔다.
    """
    new_x1 = pdf_width - x2
    new_x2 = pdf_width - x1
    new_y1 = y1
    new_y2 = y2

    # x1 < x2, y1 < y2 정렬
    new_x1, new_x2 = sorted([new_x1, new_x2])
    return (new_x1, new_y1, new_x2, new_y2)

##########################################
# ★ 항상 180도 회전 후 좌우 반전까지 ★
##########################################
FORCE_180_ROTATION = True
FORCE_HORIZONTAL_FLIP = True

for page_key, blocks in yolo_results.items():
    try:
        # page_key 예: "page_1" → 1
        page_num = int(page_key.split('_')[1])
    except Exception as e:
        print(f"페이지 키 파싱 오류: {page_key} - {e}")
        continue

    if page_num > len(doc):
        print(f"⚠️ 페이지 번호 {page_num}가 PDF 페이지 수({len(doc)})를 초과합니다!")
        continue

    page = doc[page_num - 1]
    pdf_width = page.rect.width
    pdf_height = page.rect.height

    if page_num > len(images):
        print(f"⚠️ 페이지 번호 {page_num}가 이미지 수({len(images)})를 초과합니다!")
        continue

    # pdf2image 결과 (픽셀 단위)
    pil_img = images[page_num - 1]
    img_width, img_height = pil_img.size

    # 스케일 팩터: 이미지 픽셀 / PDF 포인트
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    print(f"[Page {page_num}] PDF=({pdf_width:.2f}, {pdf_height:.2f}), "
        f"Image=({img_width}, {img_height}), scale=({scale_x:.2f}, {scale_y:.2f})")

    for block in blocks:
        # 텍스트 블록만 처리 (class_id==1)
        if block["class_id"] != 1:
            continue

        # YOLO 결과: 이미지 좌표
        x1, y1, x2, y2 = block["x1"], block["y1"], block["x2"], block["y2"]

        # 1) 이미지 좌표 → PDF 좌표 변환
        pdf_x1 = x1 / scale_x
        pdf_x2 = x2 / scale_x

        pdf_y_val1 = pdf_height - (y1 / scale_y)
        pdf_y_val2 = pdf_height - (y2 / scale_y)
        pdf_y_bottom = min(pdf_y_val1, pdf_y_val2)
        pdf_y_top = max(pdf_y_val1, pdf_y_val2)

        # 2) 180도 회전
        if FORCE_180_ROTATION:
            pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top = rotate_bbox_180(
                pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top,
                pdf_width, pdf_height
            )

        # 3) 좌우 반전
        if FORCE_HORIZONTAL_FLIP:
            pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top = flip_bbox_horizontal(
                pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top,
                pdf_width, pdf_height
            )

        # 어노테이션 추가
        rect = fitz.Rect(pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top)
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=(1, 0, 0))  # 빨간색
        annot.set_border(width=1)
        annot.update()

# 변경사항 반영
doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
doc.close()
print(f"✅ 180도 회전 + 좌우 반전이 적용된 PDF를 저장했습니다: {output_pdf_path}")
