from trl import SFTTrainer, SFTConfig

def get_trainer(model, tokenizer, train_dataset, args):
    """
    Initialize the SFTTrainer with the given model, tokenizer, and datasets.
    
    Args:
        model: The model to be trained.
        tokenizer: The tokenizer for the model.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        args: Additional arguments for the trainer.
    
    Returns:
        SFTTrainer: An instance of the SFTTrainer class.
    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = None,
        args = args)

    return trainer


def get_trainer_args(args=None):
    """
    Get the training arguments for the SFTTrainer.
    
    Returns:
        dict: A dictionary of training arguments.
    """
    if not args:
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            gradient_checkpointing= True,
            packing = True, # Use packing to reduce memor
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
            deepspeed="ds_config.json"
        )
    else:
        args = SFTConfig(
            dataset_text_field = args.dataset_text_field,
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            gradient_checkpointing= True,
            # packing = True, # Use packing to reduce memor
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs,
            learning_rate = args.learning_rate,
            logging_steps = args.logging_steps,
            optim = args.optim,
            weight_decay = args.weight_decay,
            lr_scheduler_type = args.lr_scheduler_type,
            seed = args.seed,
            report_to = args.report_to,
            deepspeed=args.deepspeed,
            # torch_empty_cache_steps=1,
            save_strategy = "steps",
            save_steps = 100,
            auto_find_batch_size = True,
            output_dir = args.output_dir,
        )
    return args

def train_model(trainer):
    """
    Train the model using the SFTTrainer.
    
    Args:
        trainer: An instance of the SFTTrainer class.
    
    Returns:
        None
    """
    trainer.train(resume_from_checkpoint = True)
