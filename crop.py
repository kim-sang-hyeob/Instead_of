import json
import os
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

'''
가장 처음 실행하는 파일. 
pdf -> img 변환 후 yolo로 바운딩 박스 추출 -> sample_bounding_boxes.json에 저장

'''


#########################
# 1) 모델 다운로드 및 로드
#########################
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(model_path, task="detect")

#########################
# 2) PDF → 이미지 변환
#########################
pdf_path = r"C:\Users\xpc\Desktop\sample.pdf"
dpi = 300
print(f"[INFO] PDF를 {dpi} dpi로 변환 중...")
images = convert_from_path(pdf_path, dpi=dpi)
print(f"[INFO] 총 {len(images)} 페이지 변환 완료.")

# 결과 저장용
results = {}

# 바운딩 박스가 그려진 이미지를 저장할 폴더 (필요시 경로 변경)
output_dir = r"C:\Users\xpc\Desktop\annotated_images"
os.makedirs(output_dir, exist_ok=True)

#########################
# 3) 페이지별 YOLO 추론
#########################
for page_idx, pil_img in enumerate(images):
    page_number = page_idx + 1
    orig_w, orig_h = pil_img.size
    print(f"[INFO] 페이지 {page_number} - 원본 이미지 크기: {orig_w} x {orig_h}")

    # YOLO에 입력하기 위해 임시 파일 저장 (안 해도 되지만, 코드 간결화를 위해)
    temp_img_path = os.path.join(output_dir, f"page_{page_number}.jpg")
    pil_img.save(temp_img_path, "JPEG")

    # doclayout_yolo 추론
    det_res = model(temp_img_path, imgsz=1024, conf=0.2, device="cuda:0")

    # 이 페이지 바운딩 박스 정보 저장용
    page_detections = []

    # 바운딩 박스가 그려진 이미지를 만들기 위한 복사본
    draw_img = pil_img.copy()
    draw_obj = ImageDraw.Draw(draw_img)

    # 모델 추론 결과
    boxes = det_res[0].boxes

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 좌상단(x1,y1), 우하단(x2,y2)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])

        # 바운딩 박스 정보 JSON에 기록
        detection_info = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": conf,
            "class_id": class_id
        }
        page_detections.append(detection_info)

        # 바운딩 박스 그리기 (빨간색 테두리)
        draw_obj.rectangle([x1, y1, x2, y2], outline="red", width=5)

        # 라벨(클래스, 신뢰도) 표시
        label_text = f"Class {class_id}: {conf:.2f}"
        draw_obj.text((x1, max(0, y1 - 20)), label_text, fill="red")

    # 이 페이지의 바운딩 박스 리스트를 결과에 저장
    results[f"page_{page_number}"] = page_detections

#########################
# 4) JSON 결과 저장
#########################
json_path = os.path.join(output_dir, "sample_bounding_boxes.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"[INFO] 바운딩 박스 좌표가 JSON으로 저장되었습니다: {json_path}")