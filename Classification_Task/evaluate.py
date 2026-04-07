import json
import os
import re
import pandas as pd

# ───────────────────────── 配置参数 ─────────────────────────
INPUT_JSON_DIR = "/data/cn/llama/Classify_Output"
BASE_OUTPUT_DIR = "/data/cn/llama/Classify_Output/Summary_Results"

SHIP_CLASSES = [
    "054A", "Admiral_Gorshkov", "Arleighburke", "Asagiri", "Atago", "Austin",
    "Barge", "Bitumen", "Blueridge", "Bulk_carrier", "Cavour", "Charles_de_Gaulle",
    "Chemical_tanker", "Container_ship", "Cruise", "Enterprise", "Epf", "Firefighting",
    "Fishing_ships", "Fpso", "Freedom_class_lcs", "Fujian", "General_cargo_ship", "Hatsuyuki",
    "Heavy_load_carrier", "Hovercraft", "Hyuga", "INS_Vikrant", "Icebreaker", "Incheon",
    "Independent_class_lcs", "Kayak", "Kirov", "La_Fayette", "Liaoning", "Lng_tanker",
    "Lpg_tanker", "Medicalship", "Monohull_sailboat", "Nimitz", "Oil_products_tanker", "Osumi",
    "Passenger_cargo_ship", "Passenger_ro-ro_ship", "Passenger_ship", "Queen_Elizabeth", 
    "R33_INS_Vikramaditya", "Reefer", "Sailing_catamaran", "Sailing_trimaran", "Sanantonio",
    "Scientific_research_ship", "Shandong", "Slava", "Tarawa", "Ticonderoga", "Tugboat",
    "Usv", "Vehicles_carrier", "Wasp", "Yuting", "Yuzhao", "Others"
]

def get_true_label(filename):
    """从文件名提取真实标签，例如 054A_109.jpg -> 054A"""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.match(r"(.+)_(\d+)$", name_without_ext, re.IGNORECASE)
    raw_label = match.group(1) if match else (name_without_ext.rsplit('_', 1)[0] if "_" in name_without_ext else name_without_ext)
    
    norm_raw = raw_label.lower().replace(" ", "_").replace("-", "_")
    for c in SHIP_CLASSES:
        if c == "Others": continue
        if c.lower().replace(" ", "_").replace("-", "_") == norm_raw:
            return c
    return "Others"

def process_single_json(json_path):
    """处理单个JSON并计算指标，结果保留两位小数(%)"""
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"读取文件失败 {json_path}: {e}")
            return None

    # 初始化统计
    total_samples = 0
    correct_top1 = 0
    # 混淆矩阵: cm[true][pred]
    cm = {t: {p: 0 for p in SHIP_CLASSES} for t in SHIP_CLASSES}
    errors = []

    for item in data:
        img_name = item.get("image_name", "")
        # 修复 KeyError: 适配 'predicted_class'
        pred_label = item.get("predicted_class", "")
        
        if pred_label == "error" or not img_name:
            continue

        total_samples += 1
        true_label = get_true_label(img_name)
        
        # 确保预测标签在预定义的 SHIP_CLASSES 中，不在则归为 Others
        if pred_label not in SHIP_CLASSES:
            pred_label = "Others"

        # 更新计数
        if true_label == pred_label:
            correct_top1 += 1
        
        cm[true_label][pred_label] += 1

        # 记录错误
        if true_label != pred_label:
            errors.append({
                "image_name": img_name,
                "true_label": true_label,
                "predicted": pred_label
            })

    if total_samples == 0:
        return None

    # ───────────────────────── 指标计算 ─────────────────────────
    # 1. Top-1 Accuracy
    acc_val = (correct_top1 / total_samples) * 100

    # 2. mAcc (Mean Accuracy)
    class_accs = []
    for c in SHIP_CLASSES:
        support = sum(cm[c].values())
        if support > 0:
            class_accs.append(cm[c][c] / support)
    mAcc_val = (sum(class_accs) / len(class_accs)) * 100 if class_accs else 0.0

    # 3. F1 相关计算
    f1_list = []
    weighted_f1_sum = 0.0
    valid_f1_classes = 0

    for c in SHIP_CLASSES:
        support = sum(cm[c].values())
        if support == 0:
            continue
            
        tp = cm[c][c]
        pred_total = sum(cm[t][c] for t in SHIP_CLASSES)
        
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_list.append(f1)
        weighted_f1_sum += (f1 * support)
        valid_f1_classes += 1

    macro_f1_val = (sum(f1_list) / valid_f1_classes) * 100 if valid_f1_classes > 0 else 0.0
    weighted_f1_val = (weighted_f1_sum / total_samples) * 100

    return {
        "accuracy": round(acc_val, 2),
        "macro_f1": round(macro_f1_val, 2),
        "mAcc": round(mAcc_val, 2),
        "weighted_f1": round(weighted_f1_val, 2),
        "total_samples": total_samples,
        "errors": errors,
        "full_report": (
            f"Total Samples: {total_samples}\n"
            f"Top-1 Acc: {acc_val:.2f}%\n"
            f"Macro F1: {macro_f1_val:.2f}%\n"
            f"mAcc: {mAcc_val:.2f}%\n"
            f"Weighted F1: {weighted_f1_val:.2f}%"
        )
    }

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    summary_data = []

    json_files = [f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')]
    
    if not json_files:
        print(f"Error: 在 {INPUT_JSON_DIR} 中未找到 JSON 文件。")
        return

    for json_file in sorted(json_files):
        print(f"正在处理: {json_file}...")
        file_id = os.path.splitext(json_file)[0]
        file_output_dir = os.path.join(BASE_OUTPUT_DIR, file_id)
        os.makedirs(file_output_dir, exist_ok=True)

        res = process_single_json(os.path.join(INPUT_JSON_DIR, json_file))
        
        if res:
            # 保存错误明细
            with open(os.path.join(file_output_dir, "errors.json"), "w", encoding="utf-8") as f:
                json.dump(res['errors'], f, indent=4, ensure_ascii=False)
            
            # 保存文本报告
            with open(os.path.join(file_output_dir, "report.txt"), "w", encoding="utf-8") as f:
                f.write(res['full_report'])

            # 收集汇总表数据
            summary_data.append({
                "Model_File": file_id,
                "Samples": res['total_samples'],
                "Top-1 Acc (%)": res['accuracy'],
                "Macro F1 (%)": res['macro_f1'],
                "mAcc (%)": res['mAcc'],
                "Weighted F1 (%)": res['weighted_f1'],
                "Error_Count": len(res['errors'])
            })

    # 生成 Excel
    if summary_data:
        df = pd.DataFrame(summary_data)
        excel_path = os.path.join(BASE_OUTPUT_DIR, "overall_summary_report.xlsx")
        # 使用 xlsxwriter 或 openpyxl
        df.to_excel(excel_path, index=False)
        print(f"\n" + "="*30)
        print(f"处理完成！")
        print(f"结果目录: {BASE_OUTPUT_DIR}")
        print(f"汇总 Excel: {excel_path}")
        print("="*30)

if __name__ == "__main__":
    main()