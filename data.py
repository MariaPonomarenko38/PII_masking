from datasets import Dataset, load_dataset, concatenate_datasets
import datasets
import pandas as pd
import json

TRAINING_PROMPT = """Here is the text that contains private information.
{input}

Your task is to find all private information in the text and rewrite it, enclosing each individual private detail in <PRIVATE> and </PRIVATE> tags.
Tag each atomic private fact separately — even if multiple facts appear in the same sentence.

Example:
Input:
Quintin, a 40-year-old logistician from Converse, TX, balances curiosity with practicality, appreciating both new ideas and established methods, and maintains a unique blend of organization and flexibility in all aspects of life.
Output:
<PRIVATE>Quintin</PRIVATE>, <PRIVATE>a 40-year-old</PRIVATE> <PRIVATE>logistician</PRIVATE> from <PRIVATE>Converse, TX</PRIVATE>, balances curiosity with practicality, appreciating both new ideas and established methods, and maintains a unique blend of organization and flexibility in all aspects of life.   

Result:
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

def prepare_dataset(dataset_repo, input_field, output_field):
    dataset = load_dataset(dataset_repo)
    train_dataset = dataset["train"]

    inputs = train_dataset[input_field]
    outputs = train_dataset[output_field]
    
    train_prompt_question = prepare_instructions(inputs, outputs)

    train_prompt_question_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_prompt_question})
    )
    return train_prompt_question_dataset 

