from datasets import Dataset, load_dataset, concatenate_datasets
import datasets
import pandas as pd
import json

TRAINING_PROMPT = """

### Instruction
Your task is to find all private information in the text and rewrite it, enclosing each individual private detail in <PRIVATE> and </PRIVATE> tags.
Tag each atomic private fact separately — even if multiple facts appear in the same sentence.

### Input
{input}

### Output
{output}
"""

def prepare_instructions(inputs, outputs):
    instructions = []

    prompt_sample = TRAINING_PROMPT

    for input, output in zip(inputs, outputs):
        example = prompt_sample.format(
            input=input,
            output=output,
        )
        instructions.append(example)

    return instructions


def insert_tags(example):
    text = example["source_text"]
    masks = sorted(example["privacy_mask"], key=lambda x: x["start"], reverse=True)  # reverse to not break offsets
    for mask in masks:
        start, end = mask["start"], mask["end"]
        label = mask["label"]
        tagged = f"<PRIVATE>{text[start:end]}</PRIVATE>"
        text = text[:start] + tagged + text[end:]
    example["source_text_tagged"] = text
    return example


def prepare_dataset_implicit(dataset_repo, input_field, output_field):
    dataset = load_dataset(dataset_repo)
    train_dataset = dataset["train"]

    inputs = train_dataset[input_field]
    outputs = train_dataset[output_field]
    
    train_prompt_question = prepare_instructions(inputs, outputs)

    train_prompt_question_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_prompt_question})
    )
    return train_prompt_question_dataset 

def prepare_dataset_explicit(dataset_repo, split, input_field):
    dataset = load_dataset(dataset_repo)[split].select(range(1000))
    dataset = dataset.map(insert_tags)

    inputs = dataset[input_field]
    outputs = dataset["source_text_tagged"]
    
    train_prompt_question = prepare_instructions(inputs, outputs)

    train_prompt_question_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_prompt_question})
    )
    return train_prompt_question_dataset 
