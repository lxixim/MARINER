import os
import json
import torch
import gc
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# ================== Configuration Parameters ==================
# VRAM Optimization: Enable expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
BASE_MODEL_PATH = "/data/cn/llama/model/Qwen2.5-VL-7B-Instruct"
LORA_CHECKPOINT_PATH = "/data/cn/llama/Classify_Output/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_lora"

TEST_IMAGE_DIR = "/data/cn/llama/classify_all/test/test"
TEST_JSON_PATH = "/data/cn/llama/classify_all/test/test.json"
OUTPUT_RESULT_PATH = "/data/cn/llama/Classify_Output/predictions.json"

MAX_NEW_TOKENS = 32
DEVICE = "cuda:0"

# ================== Class List and Prompt ==================
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

# ================== Load and Merge Model ==================
print("Step 1: Loading Base Model to CPU (to save GPU VRAM during merging)...")
# Load base model to CPU memory first to avoid VRAM contention during LoRA loading
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Critical: Keep in system RAM
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("Step 2: Loading LoRA Adapter and Merging...")
# Perform merging on CPU
model = PeftModel.from_pretrained(
    base_model, 
    LORA_CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

print("Step 3: Moving Merged Model to GPU...")
# After merging, move the entire model to GPU. It behaves as a standard 7B model (~15GB VRAM).
model.to(DEVICE) 
model.eval()

print("Step 4: Loading Processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
# For classification tasks, limiting pixels significantly reduces VRAM growth during inference
processor.image_processor.max_pixels = 512 * 28 * 28 

print("Model is ready on GPU!")

# ================== Load Test Data ==================
with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
if isinstance(test_data, dict): test_data = [test_data]

results = []

# ================== Inference Loop ==================
with torch.no_grad():
    for item in tqdm(test_data, desc="Inference"):
        image_name = item.get("image_name")
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)
        
        if not os.path.exists(image_path): continue
            
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Construct Input
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": CLASSIFY_QUERY}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)

            # Generate Output
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            # Decode
            raw_output = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            # Post-processing
            pred = raw_output if raw_output in SHIP_CLASSES else "Others"

            results.append({
                "image_name": image_name,
                "prediction": pred,
                "raw_output": raw_output
            })

            # --- Critical: Active VRAM Recovery ---
            del inputs, generated_ids, image
            # If OOM persists, uncomment the following two lines (slightly slower but safer)
            # torch.cuda.empty_cache()
            # gc.collect()

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# ================== Save Results ==================
os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)
with open(OUTPUT_RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nDone! Results saved to {OUTPUT_RESULT_PATH}")