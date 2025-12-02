import json
import sqlite3
from pathlib import Path
from torch.utils.data import Dataset

DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

# DATA LOADER
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

        self.db_root = Path(DATA_ROOT) / "database"

    def get_schema_text(self, db_id):
        """Convert DB schema into plain text."""
        schema = self.schema_map[db_id]
        out = [f"Database: {db_id}"]
        for table, col in zip(schema["table_names_original"], schema["column_names_original"]):
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
            f"[TASK]\nGenerate SQL corresponding to the question."
        )

        return {
            "input_text": input_text,
            "target_text": sql,
            "db_id": db_id,
        }

# TRAIN
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

train_data = SpiderDataset(mode="train")
dev_data   = SpiderDataset(mode="dev")

def collate_fn(batch):
    inputs = [x["input_text"] for x in batch]
    targets = [x["target_text"] for x in batch]

    model_inputs = tokenizer(
        inputs, padding=True, truncation=True, max_length=2048, return_tensors="pt"
    )
    labels = tokenizer(
        targets, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).input_ids

    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

training_args = TrainingArguments(
    output_dir="/home/jovyan/shared/sifanli/tocr/deepseekocr_spider_finetune",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=8,
    logging_steps=20,
    save_steps=2000,
    evaluation_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    data_collator=collate_fn,
)

trainer.train()

# Evaluation
def evaluate(model, tokenizer, dataset):
    exact_total = 0
    exec_total = 0
    N = len(dataset)

    for i in range(N):
        item = dataset[i]
        inp = item["input_text"]
        gold = item["target_text"]
        db_id = item["db_id"]

        inputs = tokenizer(inp, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_length=512)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)

        if exact_match(pred, gold):
            exact_total += 1

        if exec_match(pred, gold, db_id):
            exec_total += 1

        if i % 20 == 0:
            print(f"{i}/{N} done")

    print("==== Evaluation Results ====")
    print(f"Exact Match Accuracy: {exact_total/N:.4f}")
    print(f"Execution Accuracy  : {exec_total/N:.4f}")

dev_data = SpiderDataset("dev")
evaluate(model, tokenizer, dev_data)


