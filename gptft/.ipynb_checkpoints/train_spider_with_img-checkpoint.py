from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from pathlib import Path
import torch
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ======================================================
# Dataset loader (Unchanged)
# ======================================================
DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

class SpiderDataset(Dataset):
    def __init__(self, mode="train"):
        if mode == "train":
            path = Path(DATA_ROOT) / "train_spider.json"
        elif mode == "dev":
            path = Path(DATA_ROOT) / "dev.json"
        else:
            raise ValueError("mode must be train/dev")

        tables = json.load(open(Path(DATA_ROOT) / "tables.json"))
        self.schema_map = {t["db_id"]: t for t in tables}
        self.samples = json.load(open(path))

    def get_schema_text(self, db_id):
        schema = self.schema_map[db_id]
        out = [f"Database: {db_id}"]
        for table in schema["table_names_original"]:
            out.append(f"Table: {table}")
        for t_id, col in schema["column_names_original"]:
            if t_id >= 0:
                tname = schema["table_names_original"][t_id]
                out.append(f"Column: {tname}.{col}")
        return "\n".join(out)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item["question"]
        sql = item["query"]
        db_id = item["db_id"]
        schema_text = self.get_schema_text(db_id)

        input_text = (
            f"[QUESTION]\n{question}\n\n"
            f"[SCHEMA]\n{schema_text}\n\n"
            f"[TASK]\nGenerate SQL corresponding to the question.\n\n"
            f"[SQL]\n"
        )
        
        return {
            "prompt": input_text,
            "completion": sql,
            "db_id": db_id,
        }

# ======================================================
# Training (UPDATED)
# ======================================================

model_name = "deepseek-ai/DeepSeek-OCR"

# 1. Load Tokenizer Only (No ImageProcessor needed)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# 2. Load Model
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# Detect image size from config, default to 384 if not found (standard for DeepSeek-VL)
image_size = 384
if hasattr(model.config, "vision_config"):
    image_size = getattr(model.config.vision_config, "image_size", 384)

print(f"DEBUG: Using image size {image_size} for dummy inputs.")

train_data = SpiderDataset(mode="train")
dev_data = SpiderDataset(mode="dev")

# 3. FIX: Manual Image Tensor Creation in Collate Function
# We skip the AutoImageProcessor and create the tensors directly.
def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    
    # Create Dummy Images: Batch of black images [Batch, 3, H, W]
    # DeepSeek-OCR expects normalized tensors, but for text-only training, zeros are fine.
    # We use the same dtype as the model (bfloat16).
    batch_size = len(batch)
    images_tensor = torch.zeros(
        (batch_size, 3, image_size, image_size), 
        dtype=model.dtype
    )

    for item in batch:
        prompt = item["prompt"]
        completion = item["completion"] + tokenizer.eos_token
        
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        input_ids = torch.cat([prompt_ids, completion_ids])
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100 
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "images": images_tensor # Passing manual tensor directly
    }

training_args = TrainingArguments(
    output_dir="/home/jovyan/shared/sifanli/tocr/deepseekocr_spider_finetune",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=False, # Must match model dtype if using manual tensors
    bf16=True,  # Recommended for bfloat16 models
    remove_unused_columns=False 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    data_collator=collate_fn,
)

trainer.train()

