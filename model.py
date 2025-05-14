# -*- coding: utf-8 -*-
# @Time   : 2025/5/9  
# @Author : [YourName]

# 环境安装（先执行）
# pip install torch==2.0.1+cu118 transformers==4.37.0 datasets==2.14.7 
# pip install accelerate==0.27.2 peft==0.7.1 bitsandbytes==0.41.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU序号

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch

# ========== 配置区 ==========
MODEL_PATH = "Qwen/Qwen-7B-Chat"  # 使用HuggingFace官方模型[6](@ref)
DATA_PATH = "./data/cord19_sample.json"  # 示例数据路径
OUTPUT_DIR = "./qwen7b_finetuned"

# QLoRA量化配置（网页7技术）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ========== 数据加载 ==========
def format_prompt(example):
    return f"<|im_start|>system\n你是一个生物医学专家<|im_end|>\n<|im_start|>user\n生成论文摘要：{example['text']}<|im_end|>\n<|im_start|>assistant\n{example['abstract']}<|im_end|>"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_prompt)

# ========== 模型加载 ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA参数配置（网页1+网页5优化）
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ========== 训练配置 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=200
)

# ========== 开始训练 ==========
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# ========== 保存模型 ==========
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
