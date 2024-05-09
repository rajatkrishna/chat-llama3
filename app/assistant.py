from transformers import AutoTokenizer
from flask_socketio import SocketIO
from typing import List
import openvino.runtime as ov
import os
import time
import numpy as np


class LlamaAssistant():
    def __init__(self,
                 model_dir: str,
                 device: str = 'CPU',
                 role: str = "Assistant",
                 sys_prompt: str = None,
                 max_tokens: int = 500) -> None:
        model_file = os.path.join(model_dir, "openvino_model.xml")
        self.request = ov.Core().compile_model(
            model_file, device_name=device).create_infer_request()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.role = role
        self.max_tokens = max_tokens
        self.sys_msg = dict(role="system", content=sys_prompt)

    def chat(self, msgs: List[dict], socketobj: SocketIO = None):
        messages = [self.sys_msg] + msgs
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer.encode(input_ids, return_tensors="np")
        return self.generate(input_ids, socketobj=socketobj)

    def generate(self,
                 input_ids: str,
                 socketobj: SocketIO = None) -> str:
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        output_tokens = []
        new_position_id = np.copy(position_ids[..., -1:])
        inputs = {"position_ids": position_ids}
        self.request.reset_state()
        next_beam_idx = np.array([0])

        start = time.perf_counter()
        while True:
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            inputs["beam_idx"] = next_beam_idx

            self.request.infer(inputs)

            logits = self.request.get_tensor("logits").data
            next_token = np.argmax(logits[:, -1], axis=1)[0].item()
            if next_token in self.eos_token_ids or len(output_tokens) > self.max_tokens:
                break

            output_tokens += [next_token]
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            new_position_id += 1
            inputs["position_ids"] = new_position_id
            input_ids = np.array([[next_token]], dtype=np.longlong)

            if socketobj:
                latency = time.perf_counter() - start
                socketobj.emit('tps_measure', len(output_tokens)/latency)

        return self.tokenizer.decode(output_tokens)
