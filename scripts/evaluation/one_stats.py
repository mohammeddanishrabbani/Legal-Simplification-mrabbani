import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import textstat
import language_tool_python
import torch
import statistics
import json

def compute_readability_metrics(text):
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "dale_chall": textstat.dale_chall_readability_score(text)
        }
    except:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "smog_index": None,
            "dale_chall": None
        }

def compute_fluency_errors(tool, text):
    try:
        matches = tool.check(text)
        return len(matches)
    except:
        return None

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

    if "legal_text" not in df.columns or "simplified_text" not in df.columns:
        raise ValueError("CSV must contain 'legal_text' and 'simplified_text' columns.")

   

    print("Loading grammar tool...")
    tool = language_tool_python.LanguageTool("en-US")

    # Outputs
    entailment_labels = []
    entailment_confidences = []
    fluency_errors = []

    flesch_scores = []
    fk_grades = []
    smog_scores = []
    dale_chall_scores = []
    hallucination_scores = []

    print("Running evaluations...")
    results = {

    }
    
    for i in tqdm(range(0, len(df))):
        
       

        row = df.iloc[i]
        legal_text = row["legal_text"]
        simplified_text = row["simplified_text"]
        

        # Fluency
        fluency_errors.append(compute_fluency_errors(tool, legal_text))

        # Readability
        metrics = compute_readability_metrics(legal_text)
        flesch_scores.append(metrics["flesch_reading_ease"])
        fk_grades.append(metrics["flesch_kincaid_grade"])
        smog_scores.append(metrics["smog_index"])
        dale_chall_scores.append(metrics["dale_chall"])


   
    #aggregate results
    aggregate_results = {
        "average_fluency_errors": statistics.mean(fluency_errors),
        "average_flesch_reading_ease": statistics.mean(flesch_scores),
        "average_flesch_kincaid_grade": statistics.mean(fk_grades),
        "average_smog_index": statistics.mean(smog_scores),
        "average_dale_chall_score": statistics.mean(dale_chall_scores),
    }
    print("Aggregate Results:")
    for key, value in aggregate_results.items():
        print(f"{key}: {value}")
    # Save aggregate results to json
    json_path = args.output
    with open(json_path, "w") as f:
        json.dump(aggregate_results, f)
    
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a legal text simplification dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with 'legal_text' and 'simplified_text'")
    parser.add_argument("--output", type=str, required=True, help="Output CSV with evaluation results")
    args = parser.parse_args()

    evaluate_dataset(args.input, args.output)
