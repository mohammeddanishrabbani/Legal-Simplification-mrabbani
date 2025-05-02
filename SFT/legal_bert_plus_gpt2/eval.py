import time
from SFT_trainer.config import Config
from SFT_trainer.dataset_handler import DatasetHandler

from transformers import TrainingArguments, Trainer
import torch
import logging
from SFT_trainer.evaluation import Evaluation
from transformers import AutoTokenizer, EncoderDecoderModel
logging.basicConfig(level=logging.INFO)

import os
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename=f'logs/{__file__}_{time.time()}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def evaluate(dataset):
    data_list = dataset.to_list()
        for example in tqdm(data_list):
            prompt  = "{}"
            inputs = tokenizer([prompt.format(example['legal_text']),], return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]
            simplified_text = model.generate(**inputs, 
                                             max_new_tokens = 8192, 
                                             eos_token_id=tokenizer.eos_token_id, 
                                             temperature=0.7,
                                             top_k=50,
                                             top_p=0.9,
                                            repetition_penalty=1.2,
                                            no_repeat_ngram_size=3)
            simplified_text = simplified_text[:, input_length:]
            simplified_text = tokenizer.decode(simplified_text[0], skip_special_tokens=True).strip()
            print(simplified_text)
            example["prediction"] = simplified_text
            
        dataset = dataset.from_list(data_list)

        return dataset

def main():
    # Load the configuration
    model_dir = "SFT/legal_bert_plus_gpt2"
    val_dataset = "dataset/test_data.csv"
    
    encoder_model = "nlpaueb/legal-bert-base-uncased"
    
    # Load the tokenizer
    tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_model)
    model = EncoderDecoderModel.from_pretrained("SFT/legal_bert_plus_gpt2/output/Legal_bert_plus_gpt2_final")

    dataset = load_dataset("csv", data_files=val_dataset)['train']

    dataset = dataset.select(range(100))
    eval_dataset = evaluate(dataset)
    evaluation = Evaluation()   
    
    # Get the BERT score
    bert_score = evaluation.get_bert_score(eval_dataset)
    print(f"BERT Score: {bert_score}")
    rouge_score = evaluation.get_rouge_score(eval_dataset)
    print(f"ROUGE Score: {rouge_score}")
    # GET SARI SCORE
    sari_score = evaluation.get_sari_score(eval_dataset)
    print(f"SARI Score: {sari_score}")

    halucination_score = evaluation.get_halucination_score(eval_dataset)
    print(f"Hallucination Score: {halucination_score}")



    # save the evaluation results
    json_save = {
        "BERT Score": bert_score,
        "ROUGE Score": rouge_score,
        "SARI Score": sari_score,
        "Hallucination Score": halucination_score,
    }
    with open(f"{model_dir}/evaluation/eval_results.json", "w") as f:
        json.dump(json_save, f, indent=4)


    




    

    

if __name__ == "__main__":
    main()
