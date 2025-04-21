from SFT_trainer.config import Config
from SFT_trainer.dataset_handler import DatasetHandler
from SFT_trainer.model_manager import ModelManager
import SFT_trainer.trainer
from unsloth import FastModel
import torch
import logging
from unsloth.chat_templates import train_on_responses_only
logging.basicConfig(level=logging.INFO)



def main():
    # Load the configuration
    config = Config(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        dataset_path="dataset/merged_dataset.csv",
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
    

    
    # Initialize the model manager
    logging.info("Loading model...")

    model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    # full_finetuning = True, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
    )
    model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = True,

    )   
    # # Load LoRA adapter weights
    # from peft import PeftModel

    # checkpoint_dir = "trainer_output/checkpoint-528"
    # # model = PeftModel.from_pretrained(model, checkpoint_dir)



    model_manager = ModelManager(model, tokenizer, config)

    # Initialize the dataset handler
    dataset_handler = DatasetHandler(config, tokenizer, config.chat_template)
    
    # Load the dataset
    train_dataset = dataset_handler.load_dataset()
    

    
    # Preprocess the datasets
    train_dataset = dataset_handler.preprocess_dataset(train_dataset)

    
    # GET TRAINING ARGS
    args = SFT_trainer.trainer.get_trainer_args(config)
    
    # Initialize the trainer
    trainer_instance = SFT_trainer.trainer.get_trainer(model, tokenizer, train_dataset, args)
   
    # Train the model
    try:
        trainer_instance.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        trainer_instance._save_checkpoint(model,None)
    # Save the model
    model_manager.save_model()


    

if __name__ == "__main__":
    main()
