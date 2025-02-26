import fitz  # PyMuPDF
import json
import os
from pdf2image import convert_from_path

###############################
# 1. 설정
###############################
pdf_path = r"C:\Users\xpc\Desktop\sample.pdf"
json_path = r"C:\Users\xpc\Desktop\annotated_images\sample_bounding_boxes.json"
output_dir = r"C:\Users\xpc\Desktop\output"
os.makedirs(output_dir, exist_ok=True)

###############################
# handler fucntion 
###############################
# rotation 함수
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

# pdf2image로 PDF → 이미지 변환 (300dpi)
dpi = 300
images = convert_from_path(pdf_path, dpi=dpi)
doc = fitz.open(pdf_path)

# 원하는 클래스 그룹 설정
'''
{0: 'title', 1: 'plain text', 2: 'abandoned', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
텍스트  : 0,1,4,6,7,9
이미지 : 3 + 4(figure_caption)
표 : 5 + 6(table_caption) + 7(table_footnote)
'''
TEXT_CLASSES = [0, 1 ,4 , 6 , 7 , 9]
FIGURE_CLASSES = [3 , 4]
TABLE_CLASSES = [5, 6]

# 유형별 PDF 복사본 만들기
doc_text = fitz.open()    # 텍스트 전용 PDF
doc_figure = fitz.open()  # 이미지(figure) 전용 PDF
doc_table = fitz.open()   # 표 전용 PDF

###############################
# 2. JSON 결과 읽기
###############################
with open(json_path, "r", encoding="utf-8") as f:
    yolo_results = json.load(f)


##########################################
# ★ 항상 180도 회전 후 좌우 반전까지 ★
##########################################
FORCE_180_ROTATION = True
FORCE_HORIZONTAL_FLIP = True


###############################
# 3. 페이지별 처리
###############################
for page_key, blocks in yolo_results.items():
    # page_key 예: "page_1" → [page,1]
    try:
        page_num = int(page_key.split('_')[1])
    except Exception as e:
        print(f"[WARN] 페이지 키 파싱 오류: {page_key} - {e}")
        continue

    if page_num > len(doc):
        print(f"[WARN] PDF에 {page_num} 페이지가 없음!")
        continue
    if page_num > len(images):
        print(f"[WARN] 이미지에 {page_num} 페이지가 없음!")
        continue

    # 원본 페이지(주 PDF)
    orig_page = doc[page_num - 1]
    pdf_width = orig_page.rect.width
    pdf_height = orig_page.rect.height

    # pdf2image 결과 (픽셀 단위)
    pil_img = images[page_num - 1]
    img_width, img_height = pil_img.size

    # 스케일 (이미지 → PDF)
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    print(f"[INFO] Page {page_num}: PDF=({pdf_width:.1f}, {pdf_height:.1f}), "
        f"Image=({img_width}, {img_height}), scale=({scale_x:.2f}, {scale_y:.2f})")

    # PyMuPDF 페이지 복제(append) → doc_text, doc_figure, doc_table
    # (각 PDF에 페이지를 추가)
    text_page = doc_text.new_page(width=pdf_width, height=pdf_height)
    fig_page = doc_figure.new_page(width=pdf_width, height=pdf_height)
    tbl_page = doc_table.new_page(width=pdf_width, height=pdf_height)

    # 페이지에 그려줄 사각형(annot) 추가
    for block in blocks:
        x1_img, y1_img, x2_img, y2_img = block["x1"], block["y1"], block["x2"], block["y2"]
        class_id = block["class_id"]
        conf = block["confidence"]

        # 이미지 → PDF 좌표 변환
        pdf_x1 = x1_img / scale_x
        pdf_x2 = x2_img / scale_x
        pdf_y1 = pdf_height - (y1_img / scale_y)
        pdf_y2 = pdf_height - (y2_img / scale_y)
        pdf_y_bottom = min(pdf_y1, pdf_y2)
        pdf_y_top = max(pdf_y1, pdf_y2)

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

        rect = fitz.Rect(pdf_x1, pdf_y_bottom, pdf_x2, pdf_y_top)

        # 바운딩 박스 색상
        # (유형별로 다르게 설정 가능)
        color_text = (1, 0, 0)    # 빨강
        color_figure = (0, 1, 0)  # 초록
        color_table = (0, 0, 1)   # 파랑

        # 텍스트 전용 PDF
        if class_id in TEXT_CLASSES:
            annot = text_page.add_rect_annot(rect)
            annot.set_colors(stroke=color_text)
            annot.set_border(width=1)
            annot.set_info({"title": f"Text({class_id})", "content": f"conf={conf:.2f}"})
            annot.update()

        # 이미지(figure) 전용 PDF
        if class_id in FIGURE_CLASSES:
            annot = fig_page.add_rect_annot(rect)
            annot.set_colors(stroke=color_figure)
            annot.set_border(width=1)
            annot.set_info({"title": f"Figure({class_id})", "content": f"conf={conf:.2f}"})
            annot.update()

        # 표 전용 PDF
        if class_id in TABLE_CLASSES:
            annot = tbl_page.add_rect_annot(rect)
            annot.set_colors(stroke=color_table)
            annot.set_border(width=1)
            annot.set_info({"title": f"Table({class_id})", "content": f"conf={conf:.2f}"})
            annot.update()

###############################
# 4. 최종 PDF 저장
###############################
text_pdf_path = os.path.join(output_dir, "sample_text.pdf")
fig_pdf_path = os.path.join(output_dir, "sample_figure.pdf")
tbl_pdf_path = os.path.join(output_dir, "sample_table.pdf")

doc_text.save(text_pdf_path, garbage=4, deflate=True, clean=True)
doc_figure.save(fig_pdf_path, garbage=4, deflate=True, clean=True)
doc_table.save(tbl_pdf_path, garbage=4, deflate=True, clean=True)

doc_text.close()
doc_figure.close()
doc_table.close()
doc.close()

print("[DONE] 유형별 PDF 생성 완료.")
print(f" - 텍스트: {text_pdf_path}")
print(f" - 이미지: {fig_pdf_path}")
print(f" - 표: {tbl_pdf_path}")
