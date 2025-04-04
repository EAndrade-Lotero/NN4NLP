import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLM :

    def __init__(
                self, 
                model_name:str,
                safe: Optional[bool]=True
            ) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if safe:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
            )
            self.device = 'cpu'
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto"
            )
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            self.device = device

    def get_completion(
                self,
                prompt:str,
                temperature: Optional[float]=0.01
            ) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt')
        attention_mask = inputs["attention_mask"].to(self.device)
        inputs = inputs["input_ids"].to(self.device)
        output = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=True
            ),
        output = output[0][0]
        output = self.tokenizer.decode(
            output,
            skip_special_tokens=True
        )
        return output

    @staticmethod
    def from_name(model_name:str):
        return LLM(model_name)    
