from datasets import load_dataset
from config import Config
from unsloth.chat_templates import get_chat_template


class DatasetHandler:
    def __init__(self, config: Config, tokenizer, chat_template):
        self.dataset_path = config.dataset_path
        self.model_name = config.model_name
        self.output_dir = config.output_dir
        self.dataset = None
        self.dataset_split = config.dataset_split
        self.tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    def load_dataset(self):
        # Load the dataset
        self.dataset = load_dataset("csv", data_files=self.dataset_path)['train']
        return self.dataset
    
    def create_train_val_split(self):
        # Split the dataset into training and validation sets
        train_test_split = self.dataset["train"].train_test_split(test_size=self.dataset_split)
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(val_dataset)}")
        return train_dataset, val_dataset
    

    def format_chat(self, example):
        return {
            "messages": [
                {"role": "user", "content": f"### Instruction:\nYou are an expert in legal language and its interpretation. Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. The simplified version should be accessible to individuals without a legal background, using clear and concise language. ### Input:\n{example['legal_text']}"},
                {"role": "assistant", "content": f'### Response:\n {example["simplified_text"]}'},
            ]
        }
    def preprocess_dataset(self, dataset):
        if "gemma" in self.model_name:
            
            def apply_chat_template(examples):
                texts = self.tokenizer.apply_chat_template(examples["messages"])
                return { "text" : texts }
            dataset = dataset.map(self.format_chat)
            dataset = dataset.map(apply_chat_template, batched=True)
            return dataset

        if "Qwen" in self.model_name:
            EOS_TOKEN = self.tokenizer.eos_token # Must add
            def format_chat( example):
                return {
                    "text": f"""<|im_start|>system ### Instruction:\nYou are an expert in legal language and its interpretation. <|im_end|>
                    <|im_start|>user Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
                    The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
                    ### Input:\n{example['legal_text']}
                    ### Response:\n {example["simplified_text"]} <|im_end|>
                    {EOS_TOKEN}
                    """
                }

            dataset = dataset.map(format_chat)
            return dataset
        if "Llama" in self.model_name  or "mistral" in self.model_name:
            EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
            def format_chat( example):
                return {
                    "text": f"""### Instruction:\nYou are an expert in legal language and its interpretation. 
                    Your task is to simplify the following legal text while maintaining its original meaning and intent, so that a layman person can understand. 
                    The simplified version should be accessible to individuals without a legal background, using clear and concise language. 
                    ### Input:\n{example['legal_text']}
                    ### Response:\n {example["simplified_text"]}
                    {EOS_TOKEN}
                    """
                }

            dataset = dataset.map(format_chat)
            return dataset





