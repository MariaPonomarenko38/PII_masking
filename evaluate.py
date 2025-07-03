from datasets import Dataset, load_dataset, concatenate_datasets
import datasets
import pandas as pd
import json
from predict import Predictor
from tqdm import tqdm
import re
from data import prepare_dataset_explicit, prepare_dataset_implicit, insert_tags, TRAINING_PROMPT

if __name__ == "__main__":
    path = 'ponoma16/pii_detection_v2'
    path = "mistralai/Mistral-7B-Instruct-v0.3"
    predictor = Predictor(model_load_path=path)
    
    # dataset = load_dataset("ai4privacy/pii-masking-300k", split="validation").select(range(200))
    # dataset = dataset.map(insert_tags)

    dataset = load_dataset("ponoma16/implicit_pii_detection", split="validation").select(range(200))

    inputs = dataset["text"]
    outputs = dataset["annotated"]

    def extract_private_words(text):
        spans = re.findall(r"<PRIVATE>(.*?)</PRIVATE>", text, flags=re.DOTALL)
        words = []
        for span in spans:
            words += re.findall(r"\b\w+\b", span)  # extract word tokens
        return set(words)

    batch_size = 25
    total_predicted_private_words = 0
    correct_private_words = 0
    all_true_words = 0
    results = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Batches"):
        batch_inputs = inputs[i:i+batch_size]
        batch_outputs = outputs[i:i+batch_size]

        batch_prompts = [
            f"""
            ### Instruction
                Your task is to find all private information in the text and rewrite it, enclosing each individual private detail in <PRIVATE> and </PRIVATE> tags.
                Tag each atomic private fact separately — even if multiple facts appear in the same sentence.

                ### Input
                {inp}

                ### Output
            """ for inp in batch_inputs
                ]

        batch_preds = predictor.predict_batch(batch_prompts)

        for inp, true_output, pred in tqdm(
                list(zip(batch_inputs, batch_outputs, batch_preds)),
                total=len(batch_inputs),
                desc="Examples in batch",
                leave=False
            ):
            pred_words = extract_private_words(pred)
            true_words = extract_private_words(true_output)

            total_predicted_private_words += len(pred_words)
            correct_private_words += len(pred_words & true_words)
            all_true_words += len(true_words)

            results.append({
                "input": inp,
                "predicted": pred,
                "ground_truth": true_output,
                "predicted_PRIVATE_words": list(pred_words),
                "ground_truth_PRIVATE_words": list(true_words),
                "correct_PRIVATE_words": list(pred_words & true_words),
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv("private_word_eval.csv", index=False)

    if total_predicted_private_words > 0:
        precision = correct_private_words / all_true_words
        print(f"Precision of PRIVATE word prediction: {precision:.3f}")
    else:
        print("No PRIVATE words predicted.")


    