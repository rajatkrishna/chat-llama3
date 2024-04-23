from typing import List
from transformers import AutoTokenizer
import openvino.runtime as ov
import os
import time
import numpy as np


class Llama():
    def __init__(self,
                 model_path: str,
                 device: str = 'CPU',
                 eos_token_id: List = [], 
                 role: str = "Assistant") -> None:
        model_file = os.path.join(model_path, "openvino_model.xml")
        self.request = ov.Core().compile_model(
            model_file, device_name=device).create_infer_request()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_ids = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
        self.role = role

    def generate(self,
                 input: str,
                 max_generated_tokens: int = 100,
                 verbose: bool = False):
        input_ids = self.tokenizer.encode(input, return_tensors="np")
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        num_iteration = 0
        latency = 0
        output_tokens = []
        new_position_id = np.copy(position_ids[..., -1:])
        inputs = {"position_ids": position_ids}
        self.request.reset_state()
        next_beam_idx = np.array([0])

        print(f"{self.role}: ", end="", flush=True)
        while True:
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            inputs["beam_idx"] = next_beam_idx

            before = time.perf_counter()
            self.request.infer(inputs)
            after = time.perf_counter()

            if num_iteration == 0:
                first_latency = after - before
            else:
                latency += after - before
            num_iteration += 1

            logits = self.request.get_tensor("logits").data
            next_token = np.argmax(logits[:, -1], axis=1)[0].item()
            if next_token in self.eos_token_ids or len(output_tokens) > max_generated_tokens:
                break

            op_token = self.tokenizer.decode(next_token)
            print(op_token, end="", flush=True)

            output_tokens += [next_token]
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            new_position_id += 1
            inputs["position_ids"] = new_position_id
            input_ids = np.array([[next_token]], dtype=np.longlong)
        print()
        if verbose:
            print(f"{'-'*50}")
            print(f"Time to First Token: {first_latency:.2f}s")
            print(f"Total Latency: {latency:.2f}s")
            print(f"{len(output_tokens) / latency:.2f} tokens/second")
        return output_tokens, first_latency, latency
