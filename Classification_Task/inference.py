import os
import json
import torch
import gc
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# ================== 配置参数 ==================
# 显存优化：启用扩展段分配模式，减少碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
BASE_MODEL_PATH = "/data/cn/llama/model/Qwen2.5-VL-7B-Instruct"
LORA_CHECKPOINT_PATH = "/data/cn/llama/Classify_Output/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_lora"

TEST_IMAGE_DIR = "/data/cn/llama/classify_all/test/test"
TEST_JSON_PATH = "/data/cn/llama/classify_all/test/test.json"
OUTPUT_RESULT_PATH = "/data/cn/llama/Classify_Output/predictions.json"

MAX_NEW_TOKENS = 32
DEVICE = "cuda:0"

# ================== 类别列表与 Prompt (同前) ==================
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
CLASSIFY_QUERY = f"Select ONE category from the following ship types: \n{CLASSES_STR}\n\n<|image|>"

# ================== 加载并合并模型 ==================
print("Step 1: Loading Base Model to CPU (to save GPU VRAM during merging)...")
# 先将基础模型加载到 CPU 内存中，避免与 LoRA 加载过程竞争显存
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 关键：先放在内存
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("Step 2: Loading LoRA Adapter and Merging...")
# 在 CPU 上完成合并
model = PeftModel.from_pretrained(
    base_model, 
    LORA_CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    # 如果 PEFT 版本较新，可以加上这行，防止强制转 float32
    # autocast_adapter_dtype=False 
)
model = model.merge_and_unload()

print("Step 3: Moving Merged Model to GPU...")
# 合并完成后，再整体搬运到 GPU。此时它就是一个普通的 7B 模型，仅占 ~15GB 显存
model.to(DEVICE) 
model.eval()

print("Step 4: Loading Processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
# 针对分类任务，限制像素可以显著降低推理时的显存增长
processor.image_processor.max_pixels = 512 * 28 * 28 

print("✅ Model is ready on GPU!")

# ================== 加载测试数据 ==================
with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
if isinstance(test_data, dict): test_data = [test_data]

results = []

# ================== 推理循环 ==================
with torch.no_grad():
    for item in tqdm(test_data, desc="Inference"):
        image_name = item.get("image_name")
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)
        
        if not os.path.exists(image_path): continue
            
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 构造输入
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": CLASSIFY_QUERY}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)

            # 生成输出
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            # 解码
            raw_output = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            # 后处理
            pred = raw_output if raw_output in SHIP_CLASSES else "Others"

            results.append({
                "image_name": image_name,
                "prediction": pred,
                "raw_output": raw_output
            })

            # --- 关键：显存主动回收 ---
            del inputs, generated_ids, image
            # 如果依然 OOM，取消下面两行的注释（会略微降低速度但更安全）
            # torch.cuda.empty_cache()
            # gc.collect()

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# ================== 保存结果 ==================
os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)
with open(OUTPUT_RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Done! Results saved to {OUTPUT_RESULT_PATH}")