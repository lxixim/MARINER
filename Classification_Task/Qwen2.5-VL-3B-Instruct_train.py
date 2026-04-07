import os
import sys

cuda_device = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--cuda_visible_devices" and i + 1 < len(sys.argv):
        cuda_device = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

import json
import torch
import warnings
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Ship Classification Training Script")
    
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct",
                        help="Model name")
    parser.add_argument("--model_root", type=str, default="/data/cn/llama/model",
                        help="Model root directory")
    parser.add_argument("--image_root", type=str, default="/data/cn/llama/classify_all/train/images",
                        help="Image directory")
    parser.add_argument("--json_root", type=str, default="/data/cn/llama/classify_all/train",
                        help="JSON directory")
    parser.add_argument("--output_root", type=str, default="/data/cn/llama/Classify_Output",
                        help="Output root directory")
    parser.add_argument("--checkpoint_root", type=str, default="/data/cn/llama/checkpoints_classify",
                        help="Checkpoint root directory")
    
    parser.add_argument("--cuda_visible_devices", type=str, default="0",
                        help="CUDA visible devices")
    
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank (recommended: 32-64 for classification)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha (recommended: 2x lora_r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ], help="LoRA target modules")
    
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs (2-4 for 10k images)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device (adjust based on GPU memory)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = per_device * accumulation)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (recommended: 1e-5-5e-5 for multimodal LoRA)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio (smaller for large datasets)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save steps (500-1000 for 10k images)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps")
    
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")
    
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", 
                        help="Disable BF16 precision")
    parser.set_defaults(bf16=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", 
                        help="Disable gradient checkpointing")
    parser.set_defaults(gradient_checkpointing=True)
    
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Training set split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

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

<|image|>"""

def load_annotations(json_root, image_root):
    samples = []
    all_annotations = []
    
    json_files = [f for f in os.listdir(json_root) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files")
    
    for json_name in tqdm(json_files, desc="Loading JSON"):
        json_path = os.path.join(json_root, json_name)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_annotations.extend(data)
            elif isinstance(data, dict):
                all_annotations.append(data)
            else:
                print(f"Warning: Skipping invalid JSON format in {json_name}")
    
    print(f"Loaded {len(all_annotations)} annotation records")
    
    no_box_count = 0
    for item in tqdm(all_annotations, desc="Processing annotations"):
        image_name = item.get("image_name", "")
        img_path = os.path.join(image_root, image_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue
        
        label = None
        crop_box = None
        
        if "grounding_info" in item and "objects" in item["grounding_info"]:
            objects = item["grounding_info"]["objects"]
            if len(objects) > 0:
                best_obj = max(objects, key=lambda x: x.get("score", 0))
                label = best_obj.get("label", None)
                
                if "box_pixel" in best_obj:
                    crop_box = best_obj["box_pixel"]
        
        if label is None:
            label = image_name.split("_")[0]
        
        if label not in SHIP_CLASSES:
            label = "Others"
        
        if crop_box is None:
            no_box_count += 1
            crop_box = "full_image"
        
        samples.append({
            "image_path": img_path,
            "image_name": image_name,
            "label": label,
            "crop_box": crop_box
        })
    
    print(f"Valid samples: {len(samples)}")
    print(f"Samples without bounding box: {no_box_count} (using full image)")
    
    class_counts = {}
    for s in samples:
        class_counts[s["label"]] = class_counts.get(s["label"], 0) + 1
    
    print("\nClass distribution (top 10):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")
    if len(class_counts) > 10:
        print(f"  ... {len(class_counts)} total classes")
    
    return samples

class ShipClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label = sample["label"]
        crop_box = sample.get("crop_box", "full_image")
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            if crop_box != "full_image" and crop_box is not None:
                if isinstance(crop_box, str) and crop_box.startswith('['):
                    crop_box = eval(crop_box)
                x1, y1, x2, y2 = crop_box
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = int(x2), int(y2)
                image = image.crop((x1, y1, x2, y2))
                
                if image.width < 10 or image.height < 10:
                    image = Image.open(img_path).convert("RGB")
                    
        except Exception as e:
            print(f"Failed to load image: {img_path}, {e}")
            image = Image.new("RGB", (224, 224), color="black")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CLASSIFY_QUERY}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        if input_ids.dim() == 0:
            labels = torch.tensor([-100], dtype=torch.long)
        else:
            if im_start_token_id < 0 or im_start_token_id >= self.processor.tokenizer.vocab_size:
                labels[:] = -100
            else:
                im_start_mask = (input_ids == im_start_token_id)
                if im_start_mask.any():
                    im_start_indices = im_start_mask.nonzero(as_tuple=True)[0]
                    assistant_start_idx = im_start_indices[-1] + 1
                    labels[:assistant_start_idx] = -100
                else:
                    labels[:] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100
        
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is None:
            image_grid_thw = torch.tensor([[1, 28, 28]], dtype=torch.long)
        else:
            if image_grid_thw.dim() == 0:
                image_grid_thw = torch.tensor([[image_grid_thw.item(), 28, 28]], dtype=torch.long)
            elif image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            if image_grid_thw.dim() != 2:
                raise ValueError(f"image_grid_thw must be 2D, got shape {image_grid_thw.shape}")
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs.get("pixel_values", None),     
            "image_grid_thw": image_grid_thw,                      
            "labels": labels
        }

class VLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        max_len = max(len(x) for x in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        pixel_values = []
        image_grid_thw = []
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        
        for i, (ids, mask, lbl) in enumerate(zip(input_ids, attention_mask, labels)):
            pad_len = max_len - len(ids)
            
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )
            
            if features[i]["pixel_values"] is not None:
                pixel_values.append(features[i]["pixel_values"])
            if features[i]["image_grid_thw"] is not None:
                image_grid_thw.append(features[i]["image_grid_thw"])
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }
        
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
            
        return batch

def train(args):
    checkpoint_dir = os.path.join(args.checkpoint_root, args.model_name)
    output_dir = os.path.join(args.output_root, args.model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    mp = os.path.join(args.model_root, args.model_name)
    if not os.path.exists(mp): 
        mp = os.path.join(args.model_root, "Qwen", args.model_name)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model path not found: {mp}")
    
    print(f"Loading model: {mp}")
    
    processor = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    
    print("Preparing dataset...")
    samples = load_annotations(args.json_root, args.image_root)
    
    if len(samples) == 0:
        print("Error: No training data found!")
        return
    
    dataset = ShipClassificationDataset(samples, processor, max_length=args.max_length)
    
    train_size = int(args.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nTraining set: {len(train_dataset)} samples ({args.train_val_split*100:.0f}%)")
    print(f"Validation set: {len(val_dataset)} samples ({(1-args.train_val_split)*100:.0f}%)")
    
    data_collator = VLDataCollator(processor)
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs}")
    
    trainer.train()
    
    lora_output = os.path.join(output_dir, f"{args.model_name}_lora")
    model.save_pretrained(lora_output)
    processor.save_pretrained(lora_output)
    
    print(f"\nTraining complete! LoRA weights saved to: {lora_output}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    stats = {
        "model_name": args.model_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "warmup_ratio": args.warmup_ratio,
        "use_crop": True,
        "seed": args.seed,
    }
    
    with open(os.path.join(output_dir, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nTraining statistics saved to: {os.path.join(output_dir, 'train_stats.json')}")

if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("Qwen2.5-VL Ship Classification Training")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Using device: CUDA {args.cuda_visible_devices}")
    print(f"LoRA Rank: {args.lora_r}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Training epochs: {args.num_train_epochs}")
    print("=" * 60)
    
    train(args)