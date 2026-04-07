import os, json, torch, warnings, re, traceback
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ───────────────────────── 配置参数 ─────────────────────────
MODEL_NAME  = "llava-onevision-qwen2-7b-ov-hf"
MODEL_ROOT  = "/data/cn/llama/model"
DATA_ROOT   = "/data/cn/llama/classify_all/test/test"
OUTPUT_ROOT = "/data/cn/llama/Classify_Output" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ───────────────────────── 63 类船舶类型限定 ─────────────────────────
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

CLASSES_STR = "\n".join([f"- {c}" for c in SHIP_CLASSES])

CLASSIFY_QUERY = f"""You are an expert in ship identification.

Select ONE category from the following 63 ship types:

{CLASSES_STR}

Task Instructions:
1. Carefully observe the hull shape, superstructure, deck layout, and maritime features.
2. If the ship matches one of the specific categories (all except "Others"), output that EXACT category name ONLY, NO numbers, NO bullet points, NO punctuation, NO explanations.
3. If the ship does NOT match any specific category (unknown type, unclear image, or other vessel), output "Others".
"""

def match_class(txt):
    """
    将模型输出映射回 63 个标准类别
    """
    raw_txt = txt.strip()
    clean_txt = re.sub(r'^(\d+\.|-|\*)\s*', '', raw_txt)
    clean_txt = clean_txt.split("\n")[0].strip()
    
    if not clean_txt:
        return "Others", "empty_output"

    norm = clean_txt.lower().replace(" ", "_").replace("-", "_")
    
    for c in SHIP_CLASSES:
        if c.lower() == norm: 
            return c, "exact"
    
    for c in SHIP_CLASSES:
        if c == "Others":
            continue
        c_norm = c.lower().replace(" ", "_").replace("-", "_")
        if re.search(r'\b' + re.escape(c_norm) + r'\b', norm):
            return c, "substring"
    
    return "Others", "fallback"

# ───────────────────────── 核心推理逻辑 ─────────────────────────
def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    mp = os.path.join(MODEL_ROOT, MODEL_NAME)
    if not os.path.exists(mp): 
        mp = os.path.join(MODEL_ROOT, "lmms-lab", MODEL_NAME)
    if not os.path.exists(mp):
        mp = "lmms-lab/llava-onevision-qwen2-7b-ov-hf"

    print(f"正在加载模型：{mp}")
    
    # 加载 processor 和 model
    processor = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        mp, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
        device_map="cuda"
    ).eval()

    imgs = sorted([f for f in os.listdir(DATA_ROOT) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    gen_kwargs = dict(
        max_new_tokens=32, 
        do_sample=False, 
        temperature=0.0,
        pad_token_id=processor.tokenizer.eos_token_id
    )

    final_output = []
    stats = {"specific": 0, "others": 0, "error": 0}
    match_stats = {"exact": 0, "substring": 0, "fallback": 0, "empty_output": 0}

    for name in tqdm(imgs, desc="推理中"):
        try:
            img_path = os.path.join(DATA_ROOT, name)
            image = Image.open(img_path).convert("RGB")

            # LLaVA-OneVision 使用 conversation 格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": CLASSIFY_QUERY}
                    ]
                }
            ]
            
            # 使用 processor 处理输入
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            
            # 【关键修复】确保 pixel_values 与模型 dtype 一致 (bfloat16)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs)
                # 解码输出，跳过输入部分
                raw_resp = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            pred_label, match_type = match_class(raw_resp)
            match_stats[match_type] += 1
            
            if pred_label == "Others":
                stats["others"] += 1
            else:
                stats["specific"] += 1

            final_output.append({
                "image_name": name,
                "predicted_class": pred_label,
                "raw_output": raw_resp.strip(),
                "match_type": match_type
            })

        except Exception as e:
            error_msg = traceback.format_exc()
            tqdm.write(f"\n处理文件 {name} 时出错：\n{error_msg}")
            stats["error"] += 1
            final_output.append({
                "image_name": name,
                "predicted_class": "error",
                "raw_output": str(e),
                "match_type": "error"
            })

    out_path = os.path.join(OUTPUT_ROOT, f"{MODEL_NAME}_classify.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"\n推理完成！共处理 {len(final_output)} 张图片，结果保存至：{out_path}")
    print(f"统计信息：{stats}")
    print(f"匹配统计：{match_stats}")

if __name__ == "__main__":
    run()