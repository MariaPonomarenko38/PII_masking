import openai
import asyncio
from datasets import load_dataset
from aiohttp import ClientSession, ClientTimeout
from typing import List
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_CONCURRENT_REQUESTS = 10

async def annotate_record(session, sem, persona, professional_persona, person_dict) -> str:
    system_prompt = f"""
        You are a privacy annotator. Given structured personal data and a descriptive paragraph, your task is to rewrite the paragraph, incorporating all known facts from the different descriptions, structured data, and enclosing each individual private detail in <PRIVATE> and </PRIVATE> tags.

        Tag each atomic private fact separately — even if multiple facts appear in the same sentence.

        Never group multiple facts into a single tag. Each fact must have its own <PRIVATE> span.
        """

    user_prompt = f"""
        General person description: {persona},
        Description of person professional profile: {professional_persona},
        Structured personal data: {person_dict}

        Output:

        Here is example:

        Input: 

        "General person description": "Quintin, a 40-year-old logistician from Converse, TX, balances curiosity with practicality, appreciating both new ideas and established methods, and maintains a unique blend of organization and flexibility in all aspects of life."
        "Description of person professional profile": "Quintin Pete Johnson, a logistician, combines his practical nature with a curiosity for new technologies, constantly seeking to optimize supply chains while remaining grounded in proven methods."
        "Structured personal data": 
                "sex": "Male",
                "age": 40,
                "marital_status": "married_present",
                "education_level": "bachelors",
                "bachelors_field": "arts_humanities",
                "occupation": "logistician",
                "city": "Converse",
                "state": "TX",
                "zipcode": "78109",
                "country": "USA"
        
        Output:

        <PRIVATE>Quintin Pete Johnson</PRIVATE>, <PRIVATE>a 40-year-old</PRIVATE> <PRIVATE>logistician</PRIVATE> from <PRIVATE>Converse, TX 78109, USA</PRIVATE>, balances curiosity with practicality, appreciating both innovative technologies and time-tested methods. With a background in <PRIVATE>bachelor’s-level arts and humanities</PRIVATE> education, he brings a thoughtful and well-rounded perspective to his work. <PRIVATE>Married</PRIVATE> and grounded, he consistently seeks to optimize supply chains while maintaining a unique blend of organization and flexibility in all aspects of life.
        """

    async with sem:  
        try:
            response = await session.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()}
                    ],
                    "temperature": 0.3
                },
                timeout=ClientTimeout(total=60),
            )

            data = await response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error processing record: {e}")
            return f"ERROR: {e}"


async def process_dataset(dataset_name="your_dataset_name", subset=None, split="train", max_records=5000):
    dataset = load_dataset(dataset_name, subset, split=split)
    dataset = dataset.select(range(min(max_records, len(dataset))))  
    dataset = dataset.select(range(3226, max_records))
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with ClientSession() as session:
        tasks = []

        for example in dataset:
            persona = example["persona"]
            professional_persona = example["professional_persona"]
            person_dict = {
                "sex": example["sex"],
                "age": example["age"],
                "marital_status": example["marital_status"],
                "education_level": example["education_level"],
                "bachelors_field": example["bachelors_field"],
                "occupation": example["occupation"],
                "city": example["city"],
                "state": example["state"],
                "zipcode": example["zipcode"],
                "country": example["country"]
            }
            task = annotate_record(session, sem, persona, professional_persona, person_dict)
            tasks.append(task)

        results = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Annotating"):
            result = await fut
            results.append(result)
        
        #results = await asyncio.gather(*tasks)

        final = []
        for ex, output in zip(dataset, results):
            final.append({
                "id": ex["uuid"],
                "annotated": output
            })

    final = pd.DataFrame(final)
    final.to_csv("annotated_dataset1.csv", index=False)

    return final



if __name__ == "__main__":
    

    results = asyncio.run(process_dataset(dataset_name="nvidia/Nemotron-Personas"))

    for item in results[:3]:
        print("=== Annotated ===")
        print(item["annotated"])
        print()