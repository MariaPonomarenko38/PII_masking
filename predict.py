from peft import PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
                            model_load_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            device_map='auto',
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_input_ids(self, prompt: str):
        
        input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).input_ids.cuda()
        return input_ids

    @torch.inference_mode()
    def predict(self, prompt: str, max_target_length: int = 500, temperature: float = 0.01) -> str:
        input_ids = self.get_input_ids(prompt)

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
        )

        generated_tokens = outputs.sequences[:, input_ids.shape[-1]:]
        prediction = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        stop_strings = ["###", "\n\n", "<|endoftext|>"]  # adjust based on your prompt style
        for stop in stop_strings:
            if stop in prediction:
                prediction = prediction.split(stop)[0].strip()
                break

        return prediction


    @torch.inference_mode()
    def predict_batch(self, prompts: list[str], max_target_length: int = 500, temperature: float = 0.01) -> list[str]:
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
        )

        # Extract continuations only
        generated_tokens = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        stop_strings = ["###", "\n\n", "<|endoftext|>"]
        cleaned_predictions = []

        for text in decoded:
            for stop in stop_strings:
                if stop in text:
                    text = text.split(stop)[0].strip()
                    break
            cleaned_predictions.append(text)

        return cleaned_predictions

    
    
if __name__ == "__main__":
    path = 'ponoma16/pii_detection_v2'
    predictor = Predictor(model_load_path=path)
    TRAINING_PROMPT = """
    ### Instruction
        Your task is to find all private information in the text and rewrite it, enclosing each individual private detail in <PRIVATE> and </PRIVATE> tags.
        Tag each atomic private fact separately — even if multiple facts appear in the same sentence.

        ### Input
        Shirley, a 76-year-old female from Tallahassee, FL 32317, USA, is a quiet, curious soul who balances practicality with tradition. A retired business-minded librarian, she combines her love for organization with a passion for preserving community history, ensuring every book finds its rightful place on the shelf and in someone's hands. With a background in bachelor’s-level business education, she finds joy in the simple pleasures of a good book and a well-tended garden. Married and thoughtful, she cherishes the tranquility of her surroundings.

        ### Output
    """
    prediction = predictor.predict(prompt=TRAINING_PROMPT)
    print(prediction)