from unsloth import FastModel
import torch


class ModelManager():
    def __init__(self, model:FastModel, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def save_model(self):
        self.model.save_pretrained(f"{self.config.output_dir}/{self.config.model_name}")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/{self.config.model_name}")

    def load_model(self, path=None):
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)

    

