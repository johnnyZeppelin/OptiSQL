# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset
# from PIL import Image
# import os
# import json

# # --- 配置 ---
# MODEL_NAME = "deepseek-ai/DeepSeek-OCR" # 假设模型在此路径
# DATA_FILE_PATH = "processed_data/deepseek_finetune_data.jsonl" # 数据处理器生成的文件
# OUTPUT_MODEL_DIR = "deepseek_ocr_finetuned"
# LORA_R = 8
# LORA_ALPHA = 16
# BATCH_SIZE = 2
# GRADIENT_ACCUMULATION_STEPS = 4

# # --- 1. 数据集加载与格式化 ---

# class DeepSeekDataset(torch.utils.data.Dataset):
#     """
#     自定义数据集，用于加载图像、提示和目标输出。
#     注意：此处用 PIL.Image.new 模拟图像加载，实际需加载渲染的 ER 图。
#     """
#     def __init__(self, data_file):
#         self.data = [json.loads(line) for line in open(data_file, 'r')]
        
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         # **关键步骤：加载图像**
#         # 在实际操作中，您应该加载 item['image_path'] 对应的渲染后的 ER 图。
#         # 此处我们用一个空白图像作为占位符进行演示，以避免运行时错误。
#         try:
#             image = Image.open(item['image_path']).convert("RGB")
#         except FileNotFoundError:
#             # 模拟加载，实际需确保图像存在
#             image = Image.new('RGB', (512, 512), color = 'white')
            
#         # 构造输入文本序列：[Prompt] + [Target Output]
#         # VLM 需要特殊的格式来分隔输入（图像+提示）和输出（目标文本）
#         text_input = f"<|im_start|>system\n{item['prompt']}<|im_end|>\n<|im_start|>user\n{item['target_output']}<|im_end|>"
        
#         return {
#             "text": text_input,
#             "image": image,
#             "prompt": item['prompt'],
#             "target_output": item['target_output']
#         }

# def get_data_collator(tokenizer):
#     """
#     创建数据整理器，用于将数据集项转换为批处理张量。
#     此处需深度定制，因为 DeepSeek-OCR 使用视觉 Token 和文本 Token。
#     注意：需要 DeepSeek-OCR 专用的处理器来处理图像和文本。
#     """
#     # 假设 AutoTokenizer 也可以加载对应的预处理器 (Processor)
#     # 在 DeepSeek-OCR 中，通常是 ImageProcessor 和 Tokenizer 的组合
    
#     def collate_fn(batch):
#         texts = [item['text'] for item in batch]
#         images = [item['image'] for item in batch]
        
#         # 假设 DeepSeek-OCR 的 Tokenizer/Processor 能够处理图像和文本
#         # 实际情况中，DeepSeek-OCR 可能有专用的 VisionProcessor
#         inputs = tokenizer(
#             texts,
#             images=images,
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )
#         # 为因果语言模型 (Causal LM) 准备标签
#         inputs['labels'] = inputs['input_ids'].clone()
#         return inputs
        
#     return collate_fn

# # --- 2. 模型与 LoRA 配置 ---

# def setup_model_and_peft():
#     print("--- 3. 加载 DeepSeek-OCR 模型和 Tokenizer ---")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
#     # 假设模型加载类为 AutoModelForCausalLM
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME, 
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )

#     # 针对 VLM 模型的 LORA 配置 (通常针对 Q-K-V 投影层)
#     lora_config = LoraConfig(
#         r=LORA_R,
#         lora_alpha=LORA_ALPHA,
#         target_modules=["q_proj", "k_proj", "v_proj"], # 针对 MoE 解码器中的 Attention 层
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     # 应用 LoRA
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
    
#     return model, tokenizer

# # --- 3. 训练函数 ---

# def fine_tune_deepseek_ocr():
#     # 1. 自动获取数据
#     from data_processor import download_and_process_spider
#     data_path = download_and_process_spider()
    
#     # 2. 加载模型和 PEFT
#     model, tokenizer = setup_model_and_peft()

#     # 3. 加载数据集
#     train_dataset = DeepSeekDataset(data_path)
#     data_collator = get_data_collator(tokenizer)

#     # 4. 设置训练参数
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_MODEL_DIR,
#         num_train_epochs=3,
#         per_device_train_batch_size=BATCH_SIZE,
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         learning_rate=2e-5,
#         logging_steps=10,
#         save_steps=100,
#         fp16=False,
#         bf16=True, # 假设硬件支持 BF16
#         remove_unused_columns=False, # VLM 通常需要保留额外的输入 (如 pixel_values)
#     )

#     # 5. 初始化 Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#     )

#     # 6. 开始训练
#     print("--- 4. 开始微调 ---")
#     trainer.train()
    
#     # 7. 保存最终模型
#     trainer.save_model(OUTPUT_MODEL_DIR)
#     tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
#     print(f"--- 微调完成。模型保存至 {OUTPUT_MODEL_DIR} ---")

# if __name__ == '__main__':
#     fine_tune_deepseek_ocr()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor, # 用于加载图像处理器和 Tokenizer
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
import json
from typing import List, Dict

# --- 配置 ---
MODEL_NAME = "deepseek-ai/DeepSeek-OCR" 
DATA_FILE_PATH = "processed_data/deepseek_finetune_data.jsonl" 
OUTPUT_MODEL_DIR = "/home/jovyan/shared/sifanli/tocr/deepseek_ocr_finetuned"
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 3

# --- 1. 数据集加载与格式化 ---

class DeepSeekDataset(torch.utils.data.Dataset):
    """
    自定义数据集，用于加载图像、提示和目标输出。
    """
    def __init__(self, data_file):
        try:
            with open(data_file, 'r') as f:
                self.data = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file}. Please run data_processor.py first.")
            self.data = []
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # **关键步骤：加载图像**
        # 实际操作中，应加载 item['image_path'] 对应的渲染后的 ER 图。
        try:
            # 确保图像路径存在并加载
            image = Image.open(item['image_path']).convert("RGB")
        except FileNotFoundError:
            # 警告：此处的空白图像仅用于演示，实际微调中应使用真实图像
            # 否则模型无法从图像中学习到任何信息。
            print(f"Warning: Image file not found at {item['image_path']}. Using blank image placeholder.")
            image = Image.new('RGB', (512, 512), color = 'white')
            
        # 构造输入文本序列：[Prompt] + [Target Output]
        # DeepSeek-OCR 是一个 VLM，通常使用特定的 token (如 <|im_start|>) 来组织输入和输出。
        # 此处使用常见的多轮对话/指令遵循格式。
        text_input = (
            f"<|im_start|>system\n{item['prompt']}<|im_end|>"
            f"\n<|im_start|>user\n{item['target_output']}<|im_end|>"
        )
        
        return {
            "text": text_input,
            "image": image,
            # 原始 prompt 和 target_output 可用于调试，但 collator 只需要 'text' 和 'image'
        }

def get_data_collator(processor: AutoProcessor):
    """
    创建数据整理器，用于将数据集项转换为批处理张量，使用 AutoProcessor 处理 VLM 输入。
    """
    def collate_fn(batch: List[Dict]):
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        
        # 使用 Processor 处理图像和文本，并返回 PyTorch 张量
        # padding='longest' 确保批次内的所有序列长度对齐
        inputs = processor(
            texts=texts,
            images=images,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        # 为因果语言模型 (Causal LM) 准备标签
        # 标签通常是输入序列本身，但在计算损失时会进行位移和掩码，只计算目标输出部分的损失。
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
        
    return collate_fn

# --- 2. 模型与 LoRA 配置 ---

def setup_model_and_peft():
    print("--- 3. 正在从 Hugging Face 加载 DeepSeek-OCR 模型、Processor ---")
    
    # 使用 AutoProcessor 确保加载了用于处理图像和文本的组件
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # 假设模型加载类为 AutoModelForCausalLM，因为它本质上是 MoE 解码器
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 针对 VLM 模型的 LORA 配置 (通常针对 MoE 解码器中的 Q-K-V 投影层)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj"], # 针对 Attention 层的投影矩阵
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

# --- 3. 训练函数 ---

def fine_tune_deepseek_ocr():
    # 1. 自动获取数据 (需要确保 data_processor.py 在同一目录且已运行)
    try:
        from data_processor import download_and_process_spider
        data_path = download_and_process_spider()
    except ImportError:
        print("Error: data_processor.py not found. Please ensure it's in the same directory.")
        return
    except Exception as e:
        print(f"Error during data processing: {e}")
        return
    
    # 2. 加载模型、PEFT 和 Processor
    model, processor = setup_model_and_peft()

    # 3. 加载数据集
    train_dataset = DeepSeekDataset(data_path)
    # 传入 Processor 用于数据整理
    data_collator = get_data_collator(processor) 

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True, # 推荐在支持的 GPU 上使用 bfloat16
        remove_unused_columns=False, # VLM 需要保留 Processor 生成的所有列
        report_to="none", # 可选：添加 W&B 或 TensorBoard
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer, # Trainer 内部仍然需要 tokenizer
        data_collator=data_collator,
    )

    # 6. 开始训练
    print("--- 4. 开始微调 ---")
    trainer.train()
    
    # 7. 保存最终模型
    trainer.save_model(OUTPUT_MODEL_DIR)
    # 保存 LoRA 适配器和 Processor
    processor.save_pretrained(OUTPUT_MODEL_DIR) 
    print(f"--- 微调完成。模型保存至 {OUTPUT_MODEL_DIR} ---")

if __name__ == '__main__':
    fine_tune_deepseek_ocr()

