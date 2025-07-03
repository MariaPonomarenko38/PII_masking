from datasets import Dataset, load_dataset
import pandas as pd

df = pd.read_csv("annotated_dataset_val.csv") 
df["text"] = df["annotated"].str.replace(r'<PRIVATE.*?>', '', regex=True).str.replace(r'</PRIVATE>', '', regex=True)
val_dataset = Dataset.from_pandas(df)  
existing = load_dataset("ponoma16/implicit_pii_detection")
existing["validation"] = val_dataset
existing.push_to_hub("ponoma16/implicit_pii_detection")