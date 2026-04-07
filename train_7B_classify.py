import os
import sys
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
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import transformers

from qwen_vl_utils import process_vision_info

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Fine-tuning Script")
    
    # Paths
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_root", type=str, default="/data/cn/llama/model")
    parser.add_argument("--image_root", type=str, required=True, help="Image directory")
    parser.add_argument("--json_root", type=str, required=True, help="JSON directory")
    parser.add_argument("--output_root", type=str, default="/data/cn/llama/Classify_Output")
    
    # LoRA Parameters (Heavy LoRA configuration)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8) # Effective BS = 32
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=2048)
    
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    return parser.parse_args()

# ================= 2. Dataset Construction =================
class ShipClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Observe the image carefully. What is the exact category of the primary ship?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = input_ids.clone()

        if input_ids.dim() > 0:
            im_start_mask = (input_ids == self.im_start_id)
            if im_start_mask.any():
                im_start_indices = im_start_mask.nonzero(as_tuple=True)[0]
                assistant_start_idx = im_start_indices[-1] + 1
                # Mask user instructions and image tokens in labels
                labels[:assistant_start_idx] = -100

            labels[input_ids == self.processor.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs.get("pixel_values"), 
            "image_grid_thw": inputs.get("image_grid_thw")
        }

class VLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        
        pixel_values = [f["pixel_values"] for f in features if f["pixel_values"] is not None]
        image_grid_thw = [f["image_grid_thw"] for f in features if f["image_grid_thw"] is not None]
        
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
            
        return batch

def train(args):
    model_path = os.path.join(args.model_root, args.model_name)
    output_dir = os.path.join(args.output_root, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Adjust resolution settings
    processor.image_processor.max_pixels = 512 * 28 * 28
    processor.image_processor.min_pixels = 256 * 28 * 28

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", 
    )
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Preparing Dataset...")
    samples = [{"image_path": f"{args.image_root}/test.jpg", "label": "Container_ship"}] * 100 
    train_dataset = ShipClassificationDataset(samples, processor, max_length=args.max_length)
    
    print("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,  
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False, # Essential for VLM
        dataloader_num_workers=4,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLDataCollator(processor)
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Training complete! Saving LoRA to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parse_args()
    train(args)