import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import textstat
import language_tool_python
import torch
import statistics
import json
def load_hallucination_model(model_name="vectara/hallucination_evaluation_model"):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    return model

def compute_hallucination_score(model, source, generated, device):
    confidence = model.predict([(source, generated)], )
    return confidence.item()


def evaluate_dataset(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    df =df.sample(frac=0.1, random_state=42)

    if "legal_text" not in df.columns or "simplified_text" not in df.columns:
        raise ValueError("CSV must contain 'legal_text' and 'simplified_text' columns.")
    hallucination_scores = []

    print("Running evaluations...")
    results = {

    }
    model_h = load_hallucination_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in tqdm(range(0, len(df))):
        
       

        row = df.iloc[i]
        legal_text = row["legal_text"]
        simplified_text = row["simplified_text"]
      
        
        try:
            hallucination_score = compute_hallucination_score(model_h, legal_text, simplified_text, device)
            hallucination_scores.append(hallucination_score)
        except Exception as e:
            print(f"Error in hallucination model: {e}")
            pass
   
    results["hallucination_score"] = statistics.mean(hallucination_scores) if hallucination_scores else None
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a legal text simplification dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with 'legal_text' and 'simplified_text'")
    parser.add_argument("--output", type=str, required=True, help="Output CSV with evaluation results")
    args = parser.parse_args()

    result_json = evaluate_dataset(args.input, args.output)
    with open(args.output, "w") as f:
        json.dump(result_json, f)
    print(f"Results saved to {args.output}")
