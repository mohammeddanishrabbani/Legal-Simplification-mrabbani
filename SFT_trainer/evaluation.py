from dataset_handler import DatasetHandler
from unsloth import FastLanguageModel
import evaluate
from transformers import TextStreamer

from transformers import AutoModelForSequenceClassification
from tqdm import tqdm   
import torch
import numpy as np
import gc

class Evaluation:
    def __init__(self):
        pass
        # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        #     model,
        #     max_seq_length=8192,  # Choose any for long context!
        #     load_in_4bit=True,  # 4 bit quantization to reduce memory
        #     load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        #     # full_finetuning = True, # [NEW!] We have full finetuning now!
        #     # token = "hf_...", # use one if using gated models
        # )
        # self.dataset_handler = dataset_handler
        
        # self.text_streamer = TextStreamer(self.tokenizer)


    # def simplyfy_text(self, example):
    #     """
    #     Simplify the input text using the model.
        
    #     Args:
    #         text (str): The input text to be simplified.
        
    #     Returns:
    #         str: The simplified text.
    #     """
    
    #     prompt  = "{}"
    #     inputs = self.tokenizer([prompt.format(example['prompt']),], return_tensors="pt").to(self.model.device)
    #     input_ids = inputs["input_ids"]
    #     input_length = input_ids.shape[1]
    #     breakpoint()
    #     simplified_text = self.model.generate(**inputs, max_new_tokens = 1192, eos_token_id=self.tokenizer.eos_token_id)
    #     simplified_text = simplified_text[:, input_length:]
    #     simplified_text = self.tokenizer.decode(simplified_text[0], skip_special_tokens=True).strip()
    #     return {"prediction": simplified_text}
    

    # def simplyfy_text(self, example):
    #     """
    #     Simplify the input text using the model.
        
    #     Args:
    #         text (str): The input text to be simplified.
        
    #     Returns:
    #         str: The simplified text.
    #     """
    
    #     prompt  = "{}"
    #     breakpoint()
    #     inputs = self.tokenizer([prompt.format(example['prompt']),], return_tensors="pt").to(self.model.device)
    #     input_ids = inputs["input_ids"]
    #     input_length = input_ids.shape[1]
    #     simplified_text = self.model.generate(**inputs, max_new_tokens = 8192, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
    #     simplified_text = simplified_text[:, input_length:]
    #     simplified_text = self.tokenizer.decode(simplified_text[0], skip_special_tokens=True).strip()
    #     del inputs
    #     del input_ids
    #     del input_length
    #     del simplified_text
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return {"prediction": simplified_text}
    

    def evaluate(self, dataset, model, tokenizer):
        """
        Evaluate the model on the given dataset.
        
        Args:
            dataset: The dataset to evaluate the model on.
        
        Returns:
            dict: The evaluation results.
        """
   
        # def simplify_text(example):
        #     """
        #     Simplify the input text using the model.
            
        #     Args:
        #         text (str): The input text to be simplified.
            
        #     Returns:
        #         str: The simplified text.
        #     """
        
        #     prompt  = "{}"
        #     inputs = tokenizer([prompt.format(example['prompt']),], return_tensors="pt").to(model.device)
        #     input_ids = inputs["input_ids"]
        #     input_length = input_ids.shape[1]
        #     simplified_text = model.generate(**inputs, max_new_tokens = 8192, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        #     simplified_text = simplified_text[:, input_length:]
        #     simplified_text = tokenizer.decode(simplified_text[0], skip_special_tokens=True).strip()
        #     example["prediction"] = simplified_text
        #     return example
        
        # dataset = dataset.map(simplify_text)
        FastLanguageModel.for_inference(model)
        data_list = dataset.to_list()
        for example in tqdm(data_list):
            prompt  = "{}"
            inputs = tokenizer([prompt.format(example['prompt']),], return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]
            simplified_text = model.generate(**inputs, max_new_tokens = 8192, eos_token_id=tokenizer.eos_token_id)
            simplified_text = simplified_text[:, input_length:]
            simplified_text = tokenizer.decode(simplified_text[0], skip_special_tokens=True).strip()
            example["prediction"] = simplified_text
            
        dataset = dataset.from_list(data_list)

        return dataset
    

    def get_bert_score(self, eval_dataset):
        """
        Calculate the BERT score for the evaluation dataset.
        
        Args:
            eval_dataset: The evaluation dataset.
        
        Returns:
            dict: The BERT score results.
        """
        # Load the BERT scorer
        bert_scorer = evaluate.load("bertscore", model_type="bert-base-uncased")
        
        precisions = []
        recalls = []
        f1s = []
        for example in eval_dataset:
            # Get the reference and prediction texts
            reference = example["simplified_text"]
            prediction = example["prediction"]
            if prediction == "":
                continue
            # Calculate the BERT score
            results = bert_scorer.compute(predictions=[prediction], references=[reference], lang="en")
            
            precisions.append(results["precision"][0])
            recalls.append(results["recall"][0])
            f1s.append(results["f1"][0])
        # Calculate the average scores
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        return {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1
        }
    
    def get_rouge_score(self, eval_dataset):
        """
        Calculate the ROUGE score for the evaluation dataset.
        Args:
            eval_dataset: The evaluation dataset.
        Returns:
            dict: The ROUGE score results.
        """ 
        # Load the ROUGE scorer
        rouge_scorer = evaluate.load("rouge")
        
        predictions = []
        references = []
        for example in eval_dataset:
            # Get the reference and prediction texts
            reference = example["simplified_text"]
            prediction = example["prediction"]
            if prediction == "":
                continue
            
            predictions.append(prediction)
            references.append(reference)

        # Calculate the ROUGE score
        results = rouge_scorer.compute(predictions=predictions, references=references)
        return results
    

    def get_sari_score(self, eval_dataset):
        sari = evaluate.load("sari")
        scores = []
        for example in eval_dataset:
            # Get the reference and prediction texts
            reference = [example["simplified_text"]]
            prediction = example["prediction"]
            if prediction == "":
                continue
            legal_text = example["legal_text"]
            # Calculate the SARI score
            results = sari.compute(predictions=[prediction], references=[reference,], sources=[legal_text])
            scores.append(results['sari'])
        avg_sari = sum(scores) / len(scores)
        return {
            "average_sari": avg_sari
        }
    

   
    def get_halucination_score(self, eval_dataset):
        """
        Calculate the hallucination score for the evaluation dataset.
        
        Args:
            eval_dataset: The evaluation dataset.
        
        Returns:
            dict: The hallucination score results.
        """
        # Load the hallucination scorer
        model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        scores = []
        try:
            for example in tqdm(eval_dataset):
                # Get the reference and prediction texts
                reference = example["prompt"]
                prediction = example["prediction"]
                if prediction == "":
                    continue

                confidence = model.predict([(reference, prediction)], )

                scores.append(confidence.item())
        except Exception as e:
            print(f"Error in calculating hallucination score: {e}")
        avg_hallucination = sum(scores) / len(scores)
        return {
            "average_hallucination_score": float(avg_hallucination)
        }           


        
    