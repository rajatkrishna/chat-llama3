from flask_socketio import SocketIO
from llama import Llama
from transformers import AutoTokenizer
from typing import List


class Assistant:

    def __init__(self,
                 model_dir: str,
                 device: str = "CPU",
                 sys_prompt: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model = Llama(model_dir=model_dir,
                           device=device,
                           eos_token_ids=eos_token_ids)
        self.sys_msg = dict(role="system", content=sys_prompt)

    def chat(self, msgs: List[dict], socketobj: SocketIO = None):
        messages = [self.sys_msg] + msgs
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="np")
        output_tokens = self.model.generate(input_ids, socketobj=socketobj)
        return self.tokenizer.decode(output_tokens)
