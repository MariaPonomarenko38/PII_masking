import torch
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
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
from data import prepare_dataset
from constants import TRAINING_CONFIG_PATH

def main(args):
    max_seq_length = 1024

    train_dataset = prepare_dataset("ponoma16/implicit_pii_detection", "text", "annotated").select(range(100)) 

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
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0,
            r=64,
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
        
        
        processing_class = tokenizer,
        
        args = SFTConfig(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 2,
            dataset_text_field = "instructions",
            max_seq_length = max_seq_length,
            warmup_steps = 10,
            num_train_epochs=2,
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
            report_to="wandb",
            seed = 3407,
    ),
    )
    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"./assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"./results.pkl", "wb") as handle:
        run_result = [
            1,
            64,
            0.1,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    
    with open(TRAINING_CONFIG_PATH, 'r') as config_file:
        args = json.load(config_file)
    
    main(args)