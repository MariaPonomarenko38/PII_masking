import torch
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets
import pickle
import json
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from data import prepare_dataset_explicit, prepare_dataset_implicit
from constants import TRAINING_CONFIG_PATH
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
print(torch.cuda.device_count())  # should be 1
print(torch.cuda.current_device())

def main(args):
    max_seq_length = 2048

    train_dataset_implicit = prepare_dataset_implicit("ponoma16/implicit_pii_detection", "text", "annotated") 
    train_dataset_explicit = prepare_dataset_explicit("ai4privacy/pii-masking-300k", "source_text")
    train_dataset = concatenate_datasets([train_dataset_implicit, train_dataset_explicit])
    train_dataset = train_dataset.shuffle(seed=42)

    split = train_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split["train"]
    val_dataset = split["test"]
    
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        use_cache=False,
        device_map={"": "cuda:0"},
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
            lora_alpha=64,
            lora_dropout=0,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"] 
        )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        processing_class = tokenizer,
        
        args = SFTConfig(
            per_device_train_batch_size = 25,
            gradient_accumulation_steps = 2,
            num_train_epochs = 1,
            dataset_text_field = "instructions",
            max_seq_length = max_seq_length,
            warmup_steps = 10,
            #num_train_epochs=1,
            output_dir="./",
            logging_dir=f"./logs",
            #max_steps = 60,
            optim="paged_adamw_32bit",
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps=1,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            eval_steps=10,  
            eval_strategy="steps",
            report_to="wandb",
            seed = 3407,
    ),
    )
    for batch in trainer.get_train_dataloader():
        print("Batch size:", batch["input_ids"].shape)
        break
    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"./assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    print("Pushing to hub...")
    trainer.model.push_to_hub("ponoma16/pii_detection_v2")
    tokenizer.push_to_hub("ponoma16/pii_detection_v2")
    # with open(f"./results.pkl", "wb") as handle:
    #     run_result = [
    #         1,
    #         64,
    #         0.1,
    #         train_loss,
    #     ]
    #     pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    
    with open(TRAINING_CONFIG_PATH, 'r') as config_file:
        args = json.load(config_file)
    
    main(args)