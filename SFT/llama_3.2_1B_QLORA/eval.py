import json
import time
from SFT_trainer.evaluation import Evaluation
import logging
from datasets import load_dataset,load_from_disk
from unsloth import FastLanguageModel

from config import Config
from dataset_handler import DatasetHandler

logging.basicConfig(level=logging.INFO)
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename=f'logs/{__file__}_{time.time()}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


import pandas as pd
#accept the type of evaluation in the command line
import argparse
parser = argparse.ArgumentParser(description='Evaluate the model')
parser.add_argument('--eval_type', type=str, default='incontext')
parser.add_argument("--shots", type=int, default=0, help="Number of shots for in-context learning")
parser.add_argument("--inferenced_path", type=str, default=None, help="Path to the inference data")
args = parser.parse_args()
prompt = pd.read_csv("Legal-Simplification/prompt.csv")
raw_text = prompt[prompt.columns[0]].tolist()
simplified_text = prompt[prompt.columns[1]].tolist()
def maka_one_shot_prompt(example):
        """
        Create a one-shot prompt for the model.
        
        Args:
            example: The input example.
        
        Returns:
            dict: The prompt for the model.
        """
        return {
                    "prompt": f"""### Instruction:\nYou are an expert in legal language and its interpretation. 
                    Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
                    The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
                    Below is an example of a complex legal text that has been simplified by legal experts, so that a layman person can understand. Follow the given tips to simplify legal text:
                    ### Input:\n{raw_text[0]}
                    ### Output:\n{simplified_text[0]}
                    Now, please simplify the following legal text
                    ### Input:\n{example['legal_text']}
                    """
                }
    
def make_two_shot_prompt(example):
    """
    Create a two-shot prompt for the model.
    
    Args:
        example: The input example.
    
    Returns:
        dict: The prompt for the model.
    """
    return {
                "prompt": f"""### Instruction:\nYou are an expert in legal language and its interpretation. 
                Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
                The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
                Below are two examples of complex legal text that have been simplified by legal experts, so that a layman person can understand. Follow the given tips to simplify legal text:
                ### Input:\n{raw_text[0]}
                ### Output:\n{simplified_text[0]}
                ### Input:\n{raw_text[1]}
                ### Output:\n{simplified_text[1]}
                Now, please simplify the following legal text
                ### Input:\n{example['legal_text']}
                """
            }

# def make_three_shot_prompt(example):
#     """
#     Create a three-shot prompt for the model.
    
#     Args:
#         example: The input example.
    
#     Returns:
#         dict: The prompt for the model.
#     """
#     return {
#                 "prompt": f"""### Instruction:\nYou are an expert in legal language and its interpretation. 
#                 Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
#                 The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
#                 Below are three examples of complex legal text that have been simplified by legal experts, so that a layman person can understand. Follow the given tips to simplify legal text:
#                 ### Input:\n{raw_text[0]}
#                 ### Output:\n{simplified_text[0]}
#                 ### Input:\n{raw_text[1]}
#                 ### Output:\n{simplified_text[1]}
#                 ### Input:\n{raw_text[2]}
#                 ### Output:\n{simplified_text[2]}
#                 Now, please simplify the following legal text
#                 ### Input:\n{example['legal_text']}
#                 """
#             }


def make_prompt(example):
    """
    Create a prompt for the model.
    
    Args:
        example: The input example.
    
    Returns:
        dict: The prompt for the model.
    """
    return {
                "prompt": f"""### Instruction:\nYou are an expert in legal language and its interpretation. 
                Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
                The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
                ### Input:\n{example['legal_text']}
                """
            }



def main():

    model_dir = "SFT/llama_3.2_1B_QLORA"
    config = Config(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        dataset_path="dataset/train_data.csv",
        output_dir=f"{model_dir}/output",
        dataset_split=0.2,
        chat_template="llama-3",
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        packing=True,
        # deepspeed="ds_config.json",
    )
    if not args.inferenced_path:
        if args.eval_type == "finetune":
            model_path = f"{model_dir}/output/unsloth/Llama-3.2-1B-Instruct-bnb-4bit_final"
        else:
            model_path = f"unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=8192,  # Choose any for long context!
            load_in_4bit=True,  # 4 bit quantization to reduce memory
            load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
            # full_finetuning = True, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )
        
        # get evaluation dataset
        dataset_handler = DatasetHandler(config, tokenizer, config.chat_template)
        
        # Load the dataset
        val_dataset=dataset_handler.load_dataset()
    
    
        if args.eval_type == "incontext":
            if args.shots == 0:
                val_dataset = val_dataset.map(make_prompt)
            elif args.shots == 1:
                val_dataset = val_dataset.map(maka_one_shot_prompt)
            elif args.shots == 2:
                val_dataset = val_dataset.map(make_two_shot_prompt)
            else:
                raise ValueError("Invalid number of shots. Please choose 0, 1, or 2.")
            
        elif args.eval_type == "finetune":
            val_dataset = val_dataset.map(make_prompt)

        eval_dataset = val_dataset
        eval_dataset = eval_dataset.select(range(0, 100))
        evaluation = Evaluation(model, tokenizer, dataset_handler)
        # Evaluate the model
        eval_dataset = evaluation.evaluate(eval_dataset)
        # Save the evaluation results
        eval_dataset.save_to_disk(f"{model_dir}/output/eval_dataset_{args.eval_type}_{args.shots}_shots")


    # Load the evaluation dataset
    eval_dataset = load_from_disk(f"{model_dir}/output/eval_dataset_{args.eval_type}_{args.shots}_shots")
    evaluation = Evaluation(None, None, None)   
    
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
    with open(f"{model_dir}/output/eval_results_{args.eval_type}_{args.shots}_shots.json", "w") as f:
        json.dump(json_save, f, indent=4)

if __name__ == "__main__":
    main()
    


