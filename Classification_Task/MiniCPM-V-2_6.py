import os, json, torch, warnings, re
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ───────────────────────── 配置参数 ─────────────────────────
MODEL_NAME  = "MiniCPM-V-2_6"
MODEL_ROOT  = "/data/cn/llama/model"
DATA_ROOT   = "/data/cn/llama/classify_all/test/test"
OUTPUT_ROOT = "/data/cn/llama/Classify_Output" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ───────────────────────── 63类船舶类型限定 ─────────────────────────
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

# ───────────────────────── 图像预处理函数 ─────────────────────────
# MiniCPM-V-2_6 使用简单的图像加载，内部会自动处理
def load_image(path):
    image = Image.open(path).convert("RGB")
    return image

def match_class(txt):
    """
    将模型输出映射回63个标准类别
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
    
    # 更严格的子串匹配（使用单词边界）
    for c in SHIP_CLASSES:
        if c == "Others":
            continue
        c_norm = c.lower().replace(" ", "_").replace("-", "_")
        # 使用单词边界匹配，避免"ship"匹配到"fishing_ships"
        if re.search(r'\b' + re.escape(c_norm) + r'\b', norm):
            return c, "substring"
    
    # 3. 无法匹配时返回 Others
    return "Others", "fallback"

# ───────────────────────── 核心推理逻辑 ─────────────────────────
def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    mp = os.path.join(MODEL_ROOT, MODEL_NAME)
    if not os.path.exists(mp): 
        mp = os.path.join(MODEL_ROOT, "OpenBMB", MODEL_NAME)
    # 如果本地没有，尝试从HuggingFace加载
    if not os.path.exists(mp):
        mp = "openbmb/MiniCPM-V-2_6"

    print(f"正在加载模型：{mp}")
    
    # MiniCPM-V-2_6 模型加载
    model = AutoModel.from_pretrained(
        mp, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
        attn_implementation='sdpa',  # 使用SDPA注意力机制
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=False)

    imgs = sorted([f for f in os.listdir(DATA_ROOT) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    # MiniCPM-V-2_6 生成参数
    gen_cls = dict(
        max_new_tokens=32, 
        do_sample=False, 
        temperature=0.0,
    )

    final_output = []
    stats = {"specific": 0, "others": 0, "error": 0}
    match_stats = {"exact": 0, "substring": 0, "fallback": 0, "empty_output": 0}

    for name in tqdm(imgs, desc="推理中"):
        try:
            img_path = os.path.join(DATA_ROOT, name)
            image = load_image(img_path)

            # MiniCPM-V-2_6 推理调用方式 - 使用msgs格式
            msgs = [{'role': 'user', 'content': [image, CLASSIFY_QUERY]}]
            
            with torch.inference_mode():
                raw_resp = model.chat(
                    image=None,  # 图像已在msgs中传递
                    msgs=msgs, 
                    tokenizer=tokenizer, 
                    **gen_cls
                )
            
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
            tqdm.write(f"处理文件 {name} 时出错：{e}")
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

    print(f"推理完成！共处理 {len(final_output)} 张图片，结果保存至：{out_path}")
    print(f"统计信息：{stats}")
    print(f"匹配统计：{match_stats}")

if __name__ == "__main__":
    run()