import os
import json
import sqlite3
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

# ============================================================
#                   CONFIG
# ============================================================

DATA_ROOT = "/path/to/spider"
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # change if needed

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
#             LOAD SCHEMA & JSON EXAMPLES
# ============================================================

print("Loading Spider metadata...")

with open(f"{DATA_ROOT}/tables.json", "r", encoding="utf8") as f:
    schema_json = json.load(f)

schema_map = { db["db_id"]: db for db in schema_json }

def load_split(name):
    return json.load(open(f"{DATA_ROOT}/{name}.json", "r", encoding="utf8"))

train_data = load_split("train_spider") + load_split("train_others")
dev_data   = load_split("dev")

# ============================================================
#          UTIL: SAMPLE ROWS FROM SQLITE TABLE
# ============================================================

def sample_rows(db_id, table_name, k=3):
    db_path = f"{DATA_ROOT}/database/{db_id}/{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT {k}')
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    except:
        cols, rows = [], []

    conn.close()
    return cols, rows

# ============================================================
#        RENDER A TABLE + SCHEMA INTO A SIMPLE IMAGE
# ============================================================

def render_schema_image(schema, table_samples):
    W, H = 1000, 1400
    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 22)

    y = 20
    for table_name in schema["table_names"]:
        draw.text((20, y), f"Table: {table_name}", fill=(0,0,0), font=font)
        y += 30

        # Draw samples
        cols, rows = table_samples.get(table_name, ([], []))
        if cols:
            draw.text((40, y), " | ".join(cols), fill=(0,0,0), font=font)
            y += 30
            for r in rows:
                draw.text((40, y), " | ".join(map(str, r)), fill=(0,0,0), font=font)
                y += 30
        y += 20

        if y > H - 200:
            break

    return img

# ============================================================
#                   DATASET CLASS
# ============================================================

class SpiderVLMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        db_id = ex["db_id"]
        schema = schema_map[db_id]

        # sample rows for rendering
        table_samples = {}
        for t in schema["table_names"]:
            cols, rows = sample_rows(db_id, t, k=3)
            table_samples[t] = (cols, rows)

        image = render_schema_image(schema, table_samples)

        # text prompt for SQL generation
        question = ex["question"]
        sql      = ex["query"]            # gold SQL

        prompt = (
            "Given the following database tables (shown in the image) "
            "answer the natural language question by writing a valid SQL query.\n\n"
            f"Question: {question}\n"
            "SQL:"
        )

        text = prompt + " " + sql

        # DeepSeek-OCR uses processor(image, text)
        inputs = self.processor(images=image, text=text, return_tensors="pt")

        # flatten batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

# ============================================================
#               LOAD MODEL + PREPARE LORA
# ============================================================

from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============================================================
#                TRAINING SETUP
# ============================================================

train_dataset = SpiderVLMDataset(train_data)
dev_dataset   = SpiderVLMDataset(dev_data)

args = TrainingArguments(
    output_dir="./deepseek_ocr_spider_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=20,
    save_strategy="epoch",
    fp16=True if device=="cuda" else False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()
trainer.save_model("./deepseek_ocr_spider_lora")
