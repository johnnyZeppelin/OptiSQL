# # # # from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# # # # from pathlib import Path
# # # # import torch
# # # # import json
# # # # from torch.utils.data import Dataset

# # # # # ======================================================
# # # # # Dataset loader (Unchanged)
# # # # # ======================================================
# # # # DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

# # # # class SpiderDataset(Dataset):
# # # #     def __init__(self, mode="train"):
# # # #         if mode == "train":
# # # #             path = Path(DATA_ROOT) / "train_spider.json"
# # # #         elif mode == "dev":
# # # #             path = Path(DATA_ROOT) / "dev.json"
# # # #         else:
# # # #             raise ValueError("mode must be train/dev")

# # # #         tables = json.load(open(Path(DATA_ROOT) / "tables.json"))
# # # #         self.schema_map = {t["db_id"]: t for t in tables}
# # # #         self.samples = json.load(open(path))

# # # #     def get_schema_text(self, db_id):
# # # #         schema = self.schema_map[db_id]
# # # #         out = [f"Database: {db_id}"]
# # # #         for table in schema["table_names_original"]:
# # # #             out.append(f"Table: {table}")
# # # #         for t_id, col in schema["column_names_original"]:
# # # #             if t_id >= 0:
# # # #                 tname = schema["table_names_original"][t_id]
# # # #                 out.append(f"Column: {tname}.{col}")
# # # #         return "\n".join(out)

# # # #     def __len__(self):
# # # #         return len(self.samples)

# # # #     def __getitem__(self, idx):
# # # #         item = self.samples[idx]
# # # #         question = item["question"]
# # # #         sql = item["query"]
# # # #         db_id = item["db_id"]
# # # #         schema_text = self.get_schema_text(db_id)

# # # #         # Standard prompt format for Causal LM
# # # #         input_text = (
# # # #             f"[QUESTION]\n{question}\n\n"
# # # #             f"[SCHEMA]\n{schema_text}\n\n"
# # # #             f"[TASK]\nGenerate SQL corresponding to the question.\n\n"
# # # #             f"[SQL]\n" 
# # # #         )

# # # #         return {
# # # #             "input_text": input_text,
# # # #             "target_text": sql,
# # # #             "db_id": db_id,
# # # #         }

# # # # # ======================================================
# # # # # Training (UPDATED)
# # # # # ======================================================

# # # # model_name = "deepseek-ai/DeepSeek-OCR"

# # # # # 1. FIX: Ensure tokenizer has a pad token
# # # # tokenizer = AutoTokenizer.from_pretrained(
# # # #     model_name,
# # # #     trust_remote_code=True
# # # # )
# # # # if tokenizer.pad_token is None:
# # # #     tokenizer.pad_token = tokenizer.eos_token
# # # # tokenizer.padding_side = "right" # Important for training

# # # # # 2. FIX: Use AutoModelForCausalLM
# # # # model = AutoModelForCausalLM.from_pretrained(
# # # #     model_name,
# # # #     trust_remote_code=True,
# # # #     torch_dtype=torch.float16 # Recommended for DeepSeek to save memory
# # # # )

# # # # train_data = SpiderDataset(mode="train")
# # # # dev_data = SpiderDataset(mode="dev")

# # # # # 3. FIX: Updated Collate Function for Causal LM
# # # # # We must concatenate Input + Target, and mask the Input labels with -100
# # # # def collate_fn(batch):
# # # #     input_ids_list = []
# # # #     labels_list = []
# # # #     attention_mask_list = []

# # # #     for x in batch:
# # # #         prompt = x["input_text"]
# # # #         completion = x["target_text"] + tokenizer.eos_token # Add EOS to end

# # # #         # Tokenize separately to find where to mask
# # # #         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
# # # #         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
# # # #         # Concatenate
# # # #         input_ids = torch.cat([prompt_ids, completion_ids])
        
# # # #         # Create Labels: Copy input_ids, but set prompt part to -100 (ignore index)
# # # #         labels = input_ids.clone()
# # # #         labels[:len(prompt_ids)] = -100 

# # # #         input_ids_list.append(input_ids)
# # # #         labels_list.append(labels)

# # # #     # Pad the batch
# # # #     # We use torch.nn.utils.rnn.pad_sequence manually because lengths differ
# # # #     from torch.nn.utils.rnn import pad_sequence
    
# # # #     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
# # # #     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
# # # #     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()

# # # #     # Truncate if exceeding max length (optional but recommended)
# # # #     max_len = 2048
# # # #     if input_ids_padded.shape[1] > max_len:
# # # #         input_ids_padded = input_ids_padded[:, :max_len]
# # # #         labels_padded = labels_padded[:, :max_len]
# # # #         attention_mask = attention_mask[:, :max_len]

# # # #     return {
# # # #         "input_ids": input_ids_padded,
# # # #         "labels": labels_padded,
# # # #         "attention_mask": attention_mask
# # # #     }

# # # # training_args = TrainingArguments(
# # # #     output_dir="/home/jovyan/shared/sifanli/tocr/deepseekocr_spider_finetune",
# # # #     per_device_train_batch_size=1,
# # # #     per_device_eval_batch_size=1,
# # # #     gradient_accumulation_steps=16,
# # # #     learning_rate=2e-5,
# # # #     num_train_epochs=8,
# # # #     logging_steps=20,
# # # #     save_steps=2000,
# # # #     eval_strategy="epoch", # Updated from evaluation_strategy
# # # #     fp16=True, 
# # # #     remove_unused_columns=False # Important for custom datasets
# # # # )

# # # # trainer = Trainer(
# # # #     model=model,
# # # #     args=training_args,
# # # #     train_dataset=train_data,
# # # #     eval_dataset=dev_data,
# # # #     data_collator=collate_fn,
# # # # )

# # # # trainer.train()

# # # from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
# # # from pathlib import Path
# # # import torch
# # # import json
# # # from torch.utils.data import Dataset
# # # from torch.nn.utils.rnn import pad_sequence

# # # # ======================================================
# # # # Dataset loader
# # # # ======================================================
# # # DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

# # # class SpiderDataset(Dataset):
# # #     def __init__(self, mode="train"):
# # #         if mode == "train":
# # #             path = Path(DATA_ROOT) / "train_spider.json"
# # #         elif mode == "dev":
# # #             path = Path(DATA_ROOT) / "dev.json"
# # #         else:
# # #             raise ValueError("mode must be train/dev")

# # #         tables = json.load(open(Path(DATA_ROOT) / "tables.json"))
# # #         self.schema_map = {t["db_id"]: t for t in tables}
# # #         self.samples = json.load(open(path))

# # #     def get_schema_text(self, db_id):
# # #         schema = self.schema_map[db_id]
# # #         out = [f"Database: {db_id}"]
# # #         for table in schema["table_names_original"]:
# # #             out.append(f"Table: {table}")
# # #         for t_id, col in schema["column_names_original"]:
# # #             if t_id >= 0:
# # #                 tname = schema["table_names_original"][t_id]
# # #                 out.append(f"Column: {tname}.{col}")
# # #         return "\n".join(out)

# # #     def __len__(self):
# # #         return len(self.samples)

# # #     def __getitem__(self, idx):
# # #         item = self.samples[idx]
# # #         question = item["question"]
# # #         sql = item["query"]
# # #         db_id = item["db_id"]
# # #         schema_text = self.get_schema_text(db_id)

# # #         # Standard Prompt for Causal LM
# # #         # We format it so the model learns to complete the SQL
# # #         input_text = (
# # #             f"[QUESTION]\n{question}\n\n"
# # #             f"[SCHEMA]\n{schema_text}\n\n"
# # #             f"[TASK]\nGenerate SQL corresponding to the question.\n\n"
# # #             f"[SQL]\n"
# # #         )
        
# # #         return {
# # #             "prompt": input_text,
# # #             "completion": sql,
# # #             "db_id": db_id,
# # #         }

# # # # ======================================================
# # # # Training
# # # # ======================================================

# # # model_name = "deepseek-ai/DeepSeek-OCR"

# # # # 1. Load Tokenizer & Set Padding
# # # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # # # DeepSeek often doesn't have a default pad token, using EOS is standard
# # # if tokenizer.pad_token is None:
# # #     tokenizer.pad_token = tokenizer.eos_token
# # # tokenizer.padding_side = "right" 

# # # # 2. FIX: Use AutoModel instead of AutoModelForCausalLM
# # # # The config maps "AutoModel" -> "DeepseekOCRForCausalLM"
# # # model = AutoModel.from_pretrained(
# # #     model_name,
# # #     trust_remote_code=True,
# # #     torch_dtype=torch.bfloat16, # Use bfloat16 or float16 for memory efficiency
# # #     device_map="auto"
# # # )

# # # # Important: Ensure the model knows the pad token id if it uses it for attention masking
# # # model.config.pad_token_id = tokenizer.pad_token_id

# # # train_data = SpiderDataset(mode="train")
# # # dev_data = SpiderDataset(mode="dev")

# # # # 3. Custom Collate Function for Causal LM Training
# # # def collate_fn(batch):
# # #     input_ids_list = []
# # #     labels_list = []
    
# # #     for item in batch:
# # #         prompt = item["prompt"]
# # #         completion = item["completion"] + tokenizer.eos_token
        
# # #         # Tokenize prompt and completion separately to know lengths
# # #         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
# # #         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
# # #         # Concatenate: [Prompt + Completion]
# # #         input_ids = torch.cat([prompt_ids, completion_ids])
        
# # #         # Create Labels: [ -100 (mask prompt) ... -100, Completion IDs ]
# # #         labels = input_ids.clone()
# # #         labels[:len(prompt_ids)] = -100 # Ignore the prompt in loss calculation
        
# # #         input_ids_list.append(input_ids)
# # #         labels_list.append(labels)
        
# # #     # Dynamic padding to the longest sequence in the batch
# # #     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
# # #     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
# # #     # Create attention mask (1 for real tokens, 0 for pad tokens)
# # #     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
# # #     return {
# # #         "input_ids": input_ids_padded,
# # #         "labels": labels_padded,
# # #         "attention_mask": attention_mask
# # #     }

# # # training_args = TrainingArguments(
# # #     output_dir="./deepseekocr_spider_finetune",
# # #     per_device_train_batch_size=1,
# # #     gradient_accumulation_steps=16,
# # #     learning_rate=2e-5,
# # #     num_train_epochs=3,
# # #     logging_steps=10,
# # #     save_steps=500,
# # #     fp16=True, # or bf16=True if your GPU supports it
# # #     remove_unused_columns=False, # Essential when using custom collate_fn/Dataset
# # #     report_to="none"
# # # )

# # # trainer = Trainer(
# # #     model=model,
# # #     args=training_args,
# # #     train_dataset=train_data,
# # #     eval_dataset=dev_data,
# # #     data_collator=collate_fn,
# # # )

# # # trainer.train()

# # from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoImageProcessor
# # from pathlib import Path
# # import torch
# # import json
# # from torch.utils.data import Dataset
# # from torch.nn.utils.rnn import pad_sequence
# # from PIL import Image # Required for dummy images

# # # ======================================================
# # # Dataset loader (Unchanged)
# # # ======================================================
# # DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

# # class SpiderDataset(Dataset):
# #     def __init__(self, mode="train"):
# #         if mode == "train":
# #             path = Path(DATA_ROOT) / "train_spider.json"
# #         elif mode == "dev":
# #             path = Path(DATA_ROOT) / "dev.json"
# #         else:
# #             raise ValueError("mode must be train/dev")

# #         tables = json.load(open(Path(DATA_ROOT) / "tables.json"))
# #         self.schema_map = {t["db_id"]: t for t in tables}
# #         self.samples = json.load(open(path))

# #     def get_schema_text(self, db_id):
# #         schema = self.schema_map[db_id]
# #         out = [f"Database: {db_id}"]
# #         for table in schema["table_names_original"]:
# #             out.append(f"Table: {table}")
# #         for t_id, col in schema["column_names_original"]:
# #             if t_id >= 0:
# #                 tname = schema["table_names_original"][t_id]
# #                 out.append(f"Column: {tname}.{col}")
# #         return "\n".join(out)

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         item = self.samples[idx]
# #         question = item["question"]
# #         sql = item["query"]
# #         db_id = item["db_id"]
# #         schema_text = self.get_schema_text(db_id)

# #         input_text = (
# #             f"[QUESTION]\n{question}\n\n"
# #             f"[SCHEMA]\n{schema_text}\n\n"
# #             f"[TASK]\nGenerate SQL corresponding to the question.\n\n"
# #             f"[SQL]\n"
# #         )
        
# #         return {
# #             "prompt": input_text,
# #             "completion": sql,
# #             "db_id": db_id,
# #         }

# # # ======================================================
# # # Training (UPDATED)
# # # ======================================================

# # model_name = "deepseek-ai/DeepSeek-OCR"

# # # 1. Load Tokenizer & Image Processor
# # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.padding_side = "right" 

# # # 2. Load Model
# # model = AutoModel.from_pretrained(
# #     model_name,
# #     trust_remote_code=True,
# #     torch_dtype=torch.bfloat16, 
# #     device_map="auto"
# # )
# # model.config.pad_token_id = tokenizer.pad_token_id

# # train_data = SpiderDataset(mode="train")
# # dev_data = SpiderDataset(mode="dev")

# # # 3. FIX: Create Dummy Images in Collate Function
# # def collate_fn(batch):
# #     input_ids_list = []
# #     labels_list = []
    
# #     # Create a dummy black image (1024x1024 is standard for OCR high res, but 384 also works)
# #     # We create one dummy image per item in batch
# #     dummy_image = Image.new('RGB', (384, 384), color='black')
# #     batch_images = [dummy_image for _ in batch]

# #     # Process images to get tensors
# #     # The processor returns a dict usually containing 'pixel_values' or 'images'
# #     image_inputs = image_processor(images=batch_images, return_tensors="pt")
    
# #     # DeepSeek-OCR expects the key 'images' specifically in forward() 
# #     # The processor might return 'pixel_values', so we map it.
# #     if "pixel_values" in image_inputs:
# #         images_tensor = image_inputs["pixel_values"]
# #     else:
# #         images_tensor = image_inputs["images"]

# #     for item in batch:
# #         prompt = item["prompt"]
# #         completion = item["completion"] + tokenizer.eos_token
        
# #         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
# #         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
# #         input_ids = torch.cat([prompt_ids, completion_ids])
# #         labels = input_ids.clone()
# #         labels[:len(prompt_ids)] = -100 
        
# #         input_ids_list.append(input_ids)
# #         labels_list.append(labels)
        
# #     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
# #     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
# #     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
# #     return {
# #         "input_ids": input_ids_padded,
# #         "labels": labels_padded,
# #         "attention_mask": attention_mask,
# #         "images": images_tensor # PASSING THE DUMMY IMAGES HERE
# #     }

# # training_args = TrainingArguments(
# #     output_dir="./deepseekocr_spider_finetune",
# #     per_device_train_batch_size=1,
# #     gradient_accumulation_steps=16,
# #     learning_rate=2e-5,
# #     num_train_epochs=3,
# #     logging_steps=10,
# #     save_steps=500,
# #     fp16=True, 
# #     remove_unused_columns=False 
# # )

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_data,
# #     eval_dataset=dev_data,
# #     data_collator=collate_fn,
# # )

# # trainer.train()

# from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoImageProcessor
# from pathlib import Path
# import torch
# import json
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# from PIL import Image # Required for dummy images

# # ======================================================
# # Dataset loader (Unchanged)
# # ======================================================
# DATA_ROOT = "/home/jovyan/shared/sifanli/tocr/rawdata/spider_data"

# class SpiderDataset(Dataset):
#     def __init__(self, mode="train"):
#         if mode == "train":
#             path = Path(DATA_ROOT) / "train_spider.json"
#         elif mode == "dev":
#             path = Path(DATA_ROOT) / "dev.json"
#         else:
#             raise ValueError("mode must be train/dev")

#         tables = json.load(open(Path(DATA_ROOT) / "tables.json"))
#         self.schema_map = {t["db_id"]: t for t in tables}
#         self.samples = json.load(open(path))

#     def get_schema_text(self, db_id):
#         schema = self.schema_map[db_id]
#         out = [f"Database: {db_id}"]
#         for table in schema["table_names_original"]:
#             out.append(f"Table: {table}")
#         for t_id, col in schema["column_names_original"]:
#             if t_id >= 0:
#                 tname = schema["table_names_original"][t_id]
#                 out.append(f"Column: {tname}.{col}")
#         return "\n".join(out)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         question = item["question"]
#         sql = item["query"]
#         db_id = item["db_id"]
#         schema_text = self.get_schema_text(db_id)

#         input_text = (
#             f"[QUESTION]\n{question}\n\n"
#             f"[SCHEMA]\n{schema_text}\n\n"
#             f"[TASK]\nGenerate SQL corresponding to the question.\n\n"
#             f"[SQL]\n"
#         )
        
#         return {
#             "prompt": input_text,
#             "completion": sql,
#             "db_id": db_id,
#         }

# # ======================================================
# # Training (UPDATED)
# # ======================================================

# # model_name = "deepseek-ai/DeepSeek-OCR"

# # ... (Imports)
# from transformers import AutoTokenizer, AutoImageProcessor, AutoModel # Import necessary classes
# # ... (Other code)

# model_name = "deepseek-ai/DeepSeek-OCR"

# # 1. Load Tokenizer (Works reliably on its own)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# # 2. FIX: Load Image Processor, specifying the actual config file name
# # We explicitly tell it to look for 'processor_config.json' instead of 'preprocessor_config.json'.
# image_processor = AutoImageProcessor.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     image_processor_file="processor_config.json" # <--- THIS IS THE KEY FIX
# )

# # Set padding side on the tokenizer object
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right" 

# # 3. Load Model (AutoModel)
# model = AutoModel.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )
# model.config.pad_token_id = tokenizer.pad_token_id

# # # 1. Load Tokenizer & Image Processor
# # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.padding_side = "right" 

# # # 2. Load Model
# # model = AutoModel.from_pretrained(
# #     model_name,
# #     trust_remote_code=True,
# #     torch_dtype=torch.bfloat16, 
# #     device_map="auto"
# # )
# # model.config.pad_token_id = tokenizer.pad_token_id

# # train_data = SpiderDataset(mode="train")
# # dev_data = SpiderDataset(mode="dev")

# # # 3. FIX: Create Dummy Images in Collate Function
# # def collate_fn(batch):
# #     input_ids_list = []
# #     labels_list = []
    
# #     # Create a dummy black image (1024x1024 is standard for OCR high res, but 384 also works)
# #     # We create one dummy image per item in batch
# #     dummy_image = Image.new('RGB', (384, 384), color='black')
# #     batch_images = [dummy_image for _ in batch]

# #     # Process images to get tensors
# #     # The processor returns a dict usually containing 'pixel_values' or 'images'
# #     image_inputs = image_processor(images=batch_images, return_tensors="pt")
    
# #     # DeepSeek-OCR expects the key 'images' specifically in forward() 
# #     # The processor might return 'pixel_values', so we map it.
# #     if "pixel_values" in image_inputs:
# #         images_tensor = image_inputs["pixel_values"]
# #     else:
# #         images_tensor = image_inputs["images"]

# #     for item in batch:
# #         prompt = item["prompt"]
# #         completion = item["completion"] + tokenizer.eos_token
        
# #         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
# #         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
# #         input_ids = torch.cat([prompt_ids, completion_ids])
# #         labels = input_ids.clone()
# #         labels[:len(prompt_ids)] = -100 
        
# #         input_ids_list.append(input_ids)
# #         labels_list.append(labels)
        
# #     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
# #     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
# #     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
# #     return {
# #         "input_ids": input_ids_padded,
# #         "labels": labels_padded,
# #         "attention_mask": attention_mask,
# #         "images": images_tensor # PASSING THE DUMMY IMAGES HERE
# #     }

# # # 1. FIX: Load the combined processor instead of separate components
# # from transformers import AutoProcessor
# # processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# # # 2. Extract components from the processor object
# # tokenizer = processor.tokenizer
# # image_processor = processor.image_processor

# # # FIX: Set padding side on the tokenizer object
# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.padding_side = "right" 

# # # 3. Load Model (Using AutoModel as fixed previously)
# # model = AutoModel.from_pretrained(
# #     model_name,
# #     trust_remote_code=True,
# #     torch_dtype=torch.bfloat16, 
# #     device_map="auto"
# # )
# # model.config.pad_token_id = tokenizer.pad_token_id

# # train_data = SpiderDataset(mode="train")
# # dev_data = SpiderDataset(mode="dev")

# # # 4. Collate Function Update (Now uses the global image_processor)
# # def collate_fn(batch):
# #     input_ids_list = []
# #     labels_list = []
    
# #     # Create a dummy black image for every item in the batch
# #     dummy_image = Image.new('RGB', (384, 384), color='black')
# #     batch_images = [dummy_image for _ in batch]

# #     # Process images using the image_processor object
# #     image_inputs = image_processor(images=batch_images, return_tensors="pt")
    
# #     # Get the image tensor (DeepSeek-OCR often uses the key 'pixel_values' or 'images')
# #     images_tensor = image_inputs.get("pixel_values") or image_inputs.get("images")

# #     for item in batch:
# #         prompt = item["prompt"]
# #         completion = item["completion"] + tokenizer.eos_token
        
# #         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
# #         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
# #         input_ids = torch.cat([prompt_ids, completion_ids])
# #         labels = input_ids.clone()
# #         labels[:len(prompt_ids)] = -100 
        
# #         input_ids_list.append(input_ids)
# #         labels_list.append(labels)
        
# #     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
# #     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
# #     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
# #     return {
# #         "input_ids": input_ids_padded,
# #         "labels": labels_padded,
# #         "attention_mask": attention_mask,
# #         "images": images_tensor
# #     }

# # # 1. FIX: Load Tokenizer and Image Processor separately 
# # # This bypasses the issue where AutoProcessor returns the wrong object.
# # from transformers import AutoTokenizer, AutoImageProcessor
# # # AutoProcessor is no longer needed in the import list.

# # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

# # # Set padding side on the tokenizer object
# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.padding_side = "right" 

# # # 2. Load Model (No change from previous successful step)
# # model = AutoModel.from_pretrained(
# #     model_name,
# #     trust_remote_code=True,
# #     torch_dtype=torch.bfloat16, 
# #     device_map="auto"
# # )
# # model.config.pad_token_id = tokenizer.pad_token_id

# train_data = SpiderDataset(mode="train")
# dev_data = SpiderDataset(mode="dev")

# # 3. Collate Function (No changes needed inside, it uses the global variables)
# def collate_fn(batch):
#     input_ids_list = []
#     labels_list = []
    
#     # Create a dummy black image for every item in the batch
#     dummy_image = Image.new('RGB', (384, 384), color='black')
#     batch_images = [dummy_image for _ in batch]

#     # Process images using the image_processor object
#     image_inputs = image_processor(images=batch_images, return_tensors="pt")
    
#     # Get the image tensor 
#     # Use .get() to safely check for keys like 'pixel_values' or 'images'
#     images_tensor = image_inputs.get("pixel_values") or image_inputs.get("images")

#     for item in batch:
#         prompt = item["prompt"]
#         completion = item["completion"] + tokenizer.eos_token
        
#         prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
#         completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
#         input_ids = torch.cat([prompt_ids, completion_ids])
#         labels = input_ids.clone()
#         labels[:len(prompt_ids)] = -100 
        
#         input_ids_list.append(input_ids)
#         labels_list.append(labels)
        
#     input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
#     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
#     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
#     return {
#         "input_ids": input_ids_padded,
#         "labels": labels_padded,
#         "attention_mask": attention_mask,
#         "images": images_tensor
#     }

# training_args = TrainingArguments(
#     output_dir="./deepseekocr_spider_finetune",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=16,
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     logging_steps=10,
#     save_steps=500,
#     fp16=True, 
#     remove_unused_columns=False 
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=dev_data,
#     data_collator=collate_fn,
# )

# trainer.train()

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
    output_dir="./deepseekocr_spider_finetune",
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

