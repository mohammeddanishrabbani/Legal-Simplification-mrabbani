import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Load your dataset
df = pd.read_csv("dataset/merged_dataset.csv")
# df = df.sample(frac=0.1, random_state=42)  # Sample 10% of the dataset for evaluation
assert 'legal_text' in df.columns and 'simplified_text' in df.columns, "CSV must have 'legal_text' and 'simplified_text' columns."

# Load sentence similarity model
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load NLI model for hallucination/entailment
nli_model_name = "facebook/bart-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)

def get_score(premise, hypothesis):
    x = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt',truncation=True, padding=True)
    
    logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]
    return prob_label_is_true

def compute_cosine_similarity(s1, s2):
    emb1 = similarity_model.encode(s1, convert_to_tensor=True)
    emb2 = similarity_model.encode(s2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# Evaluation
results = []
entailment_probs = []
similarity_scores = []
semantic_count = 0
entailment_count = 0
joint_count = 0

for index, row in tqdm(df.iterrows(), total=len(df)):
    source = str(row['legal_text'])
    simplified = str(row['simplified_text'])

    sim_score = compute_cosine_similarity(source, simplified)
    similarity_scores.append(sim_score)
    score = get_score(simplified, source)
    entailment_probs.append(score.item())
    if sim_score > 0.65:
        semantic_count += 1

    if score.item() > 0.5:
        entailment_count += 1
    
    if sim_score > 0.65 and score.item() > 0.5:
        joint_count += 1
    

    


    
entailment_score = sum(entailment_probs) / len(entailment_probs) if entailment_probs else 0
print(f"Average Entailment Score: {entailment_score:.4f}")
print(f"Average Cosine Similarity: {sum(similarity_scores) / len(similarity_scores):.4f}")
print(f"Entailment Count: {entailment_count/len(df) * 100:.2f}%")
print(f"Semantic Similarity Count: {semantic_count/len(df) * 100:.2f}%")
print(f"Joint Count: {joint_count/len(df) * 100:.2f}%")

