import json
from SFT_trainer.evaluation import Evaluation

from unsloth import FastLanguageModel

from config import Config
from dataset_handler import DatasetHandler



def main():
    config = Config(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        dataset_path="dataset/test_data.csv",
        output_dir="SFT/llama_3.2_1B_QLORA/output",
        dataset_split=0.2,
        chat_template="llama-3",
        dataset_text_field="text",
        per_device_train_batch_size=1,
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
    model_path = "SFT/llama_3.2_1B_QLORA/output/unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
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
    
    eval_dataset = val_dataset.map(make_prompt)
    eval_dataset = eval_dataset.select(range(100))  # Select a subset for evaluation
    evaluation = Evaluation(model, tokenizer, dataset_handler)
    # Evaluate the model
    eval_dataset = evaluation.evaluate(eval_dataset)
    # Get the BERT score
    bert_score = evaluation.get_bert_score(eval_dataset)
    print(f"BERT Score: {bert_score}")
    rouge_score = evaluation.get_rouge_score(eval_dataset)
    print(f"ROUGE Score: {rouge_score}")
    # GET SARI SCORE
    sari_score = evaluation.get_sari_score(eval_dataset)
    print(f"SARI Score: {sari_score}")

    # save the evaluation results
    json_save = {
        "BERT Score": bert_score,
        "ROUGE Score": rouge_score,
        "SARI Score": sari_score
    }
    with open("SFT/llama_3.2_1B_QLORA/eval_results.json", "w") as f:
        json.dump(json_save, f, indent=4)

if __name__ == "__main__":
    main()
    


