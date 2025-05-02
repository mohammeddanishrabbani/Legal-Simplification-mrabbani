import time
from SFT_trainer.config import Config
from SFT_trainer.dataset_handler import DatasetHandler
from SFT_trainer.model_manager import ModelManager
import SFT_trainer.trainer

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
    decoder_model = "gpt2"  # You can use t5-small if preferred

    # Load the tokenizer
    tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_model)
    tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_model)
    tokenizer_decoder.pad_token = tokenizer_decoder.eos_token
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
    
    dataset_handler = DatasetHandler(config, tokenizer_decoder)

    # Load the dataset
    dataset = dataset_handler.load_dataset()
    # Preprocess the dataset
    dataset = dataset_handler.preprocess_dataset(dataset)
    del dataset_handler

    args = SFT_trainer.trainer.get_trainer_args(config)

    trainer_instance = SFT_trainer.trainer.get_trainer(model, tokenizer_decoder, dataset, args)

    try:
        if checkpoint_dir:
            trainer_instance.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer_instance.train()
    except Exception as e:
        logging.error(f"Error during training: {e}")
        trainer_instance._save_checkpoint(model,None)
    model.save_pretrained(f"{config.output_dir}/{config.model_name}_final")
    tokenizer_decoder.save_pretrained(f"{config.output_dir}/{config.model_name}_final")





    

    

if __name__ == "__main__":
    main()
