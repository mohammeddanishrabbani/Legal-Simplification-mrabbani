import time
from SFT_trainer.config import Config
from SFT_trainer.dataset_handler import DatasetHandler

from transformers import TrainingArguments, Trainer
import torch
import logging

from transformers import AutoTokenizer, EncoderDecoderModel
logging.basicConfig(level=logging.INFO)

import os
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename=f'logs/{__file__}_{time.time()}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')



def main():
    # Load the configuration
    model_dir = "SFT/legal_bert_plus_gpt2"
    # checkpoint_dir = f"{model_dir}/checkpoint-528"
    checkpoint_dir = None
    config = Config(
        model_name="Legal_bert_plus_gpt2",
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
    
    encoder_model = "nlpaueb/legal-bert-base-uncased"
    decoder_model = "gpt2-medium" 

    # Load the tokenizer
    tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_model)
    tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_model)
    tokenizer_decoder.pad_token = tokenizer_decoder.eos_token
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
    model.config.decoder_start_token_id = tokenizer_decoder.bos_token_id
    model.config.pad_token_id = tokenizer_decoder.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    dataset_handler = DatasetHandler(config, tokenizer_decoder)

    # Load the dataset
    dataset = dataset_handler.load_dataset()
    # Preprocess the dataset
    def preprocess(example):
        inputs = tokenizer_encoder(example["legal_text"], truncation=True, padding="max_length", max_length=512)
        targets = tokenizer_decoder(example["simplified_text"], truncation=True, padding="max_length", max_length=1024)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }
    dataset = dataset.map(preprocess)
    dataset = dataset.remove_columns(['legal_text', 'simplified_text'])
    del dataset_handler
    args = TrainingArguments(
                                output_dir=f"{model_dir}/output",
                                per_device_train_batch_size=1,
                                num_train_epochs=1,
                                save_strategy="steps",
                                save_steps=100,
                                logging_dir="./logs",
                                logging_steps=100,
                                warmup_steps=500,
                                weight_decay=0.01,
                                fp16=True,  # if using a GPU
                                )
    # Initialize the Trainer
    trainer_instance = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer_decoder,
    )

    

    try:
        if checkpoint_dir:
            trainer_instance.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer_instance.train()
    except Exception as e:
        logging.error(f"Error during training: {e}")
        model.save_pretrained(f"{config.output_dir}/{config.model_name}_checkpoint")
        tokenizer_decoder.save_pretrained(f"{config.output_dir}/{config.model_name}_checkpoint")
    
    model.save_pretrained(f"{config.output_dir}/{config.model_name}_final")
    tokenizer_decoder.save_pretrained(f"{config.output_dir}/{config.model_name}_final")





    

    

if __name__ == "__main__":
    main()
