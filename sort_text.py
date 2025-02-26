import json
import statistics

def sort_text_blocks_by_class1_basis(blocks, x_threshold=200.0):
    """
    모든 텍스트 블록(모든 class 대상)을 읽기 순서대로 정렬합니다.
    
    다중 컬럼 여부는 오직 class 1 블록들의 x_center 분포를 기준으로 결정합니다.
    1. 먼저, class 1 블록들의 중심 x 좌표(x_center)를 계산하여 분포(spread)와 중앙값(median_x)를 구합니다.
    2. 만약 class 1 블록들의 x_center spread가 x_threshold보다 크면 다중 컬럼으로 간주하고,
       모든 블록(모든 class)을 대상으로 각 블록의 x_center가 중앙값보다 작으면 왼쪽, 크면 오른쪽 컬럼으로 분리합니다.
       각 컬럼 내에서는 bbox의 하단 좌표(y1, 즉 페이지 하단에 가까운 값) 오름차순, 그 후 x1 오름차순으로 정렬합니다.
    3. 그렇지 않으면 단일 컬럼으로 간주하여 모든 블록을 (y1, x1) 오름차순으로 정렬합니다.
    
    PDF 좌표계(원점: 왼쪽 하단)를 기준으로 하므로, y1 값이 작을수록 페이지 하단에 위치합니다.
    """
    if not blocks:
        return blocks

    # 모든 블록에 대해 x_center, x1, y1 계산
    for block in blocks:
        x1, y1, x2, y2 = block["pdf_bbox"]
        block["x_center"] = (x1 + x2) / 2
        block["x1_val"] = x1
        block["y1_val"] = y1  # 하단 좌표

    # class 1 블록들만 기준으로 다중 컬럼 여부 판단
    class1_blocks = [b for b in blocks if b.get("class_id") == 1]
    if class1_blocks:
        class1_x_centers = [b["x_center"] for b in class1_blocks]
        spread = max(class1_x_centers) - min(class1_x_centers)
        median_x = statistics.median(class1_x_centers)
    else:
        # class 1 블록이 없으면 모든 블록을 대상으로 함
        class1_x_centers = [b["x_center"] for b in blocks]
        spread = max(class1_x_centers) - min(class1_x_centers)
        median_x = statistics.median(class1_x_centers)

    if spread > x_threshold:
        # 다중 컬럼으로 간주
        left_column = [b for b in blocks if b["x_center"] <= median_x]
        right_column = [b for b in blocks if b["x_center"] > median_x]
        # 각 컬럼 내에서 y1 오름차순, 그 후 x1 오름차순 정렬 (즉, 페이지 하단부터 위쪽으로, 같은 행에서는 좌측부터)
        left_sorted = sorted(left_column, key=lambda b: (b["y1_val"], b["x1_val"]))
        right_sorted = sorted(right_column, key=lambda b: (b["y1_val"], b["x1_val"]))
        sorted_blocks = left_sorted + right_sorted
    else:
        # 단일 컬럼으로 간주
        sorted_blocks = sorted(blocks, key=lambda b: (b["y1_val"], b["x1_val"]))

    # 임시 키 제거
    for block in blocks:
        block.pop("x_center", None)
        block.pop("x1_val", None)
        block.pop("y1_val", None)

    return sorted_blocks

if __name__ == "__main__":
    # 페이지별 구조를 유지한 기존 JSON 파일을 로드 (예: structured_data.json)
    with open(r"C:\Users\xpc\Desktop\output\structured_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    sorted_data = {}
    # 각 페이지별로 text_blocks 배열만 정렬 (모든 클래스 대상)
    for page, content in data.items():
        text_blocks = content.get("text_blocks", [])
        sorted_text = sort_text_blocks_by_class1_basis(text_blocks, x_threshold=200.0)
        new_content = content.copy()
        new_content["text_blocks"] = sorted_text
        sorted_data[page] = new_content

    # 결과를 새 JSON 파일로 저장
    with open("sorted_structured_data_class1_based.json", "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)

    print("페이지별 텍스트 블록 정렬 완료 (기준: class 1). 결과는 'sorted_structured_data_class1_based.json'에 저장되었습니다.")
