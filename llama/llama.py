from flask_socketio import SocketIO
import openvino.runtime as ov
import os
import time
import numpy as np
from typing import List


class Llama():

    def __init__(self,
                 model_dir: str,
                 device: str = "CPU",
                 eos_token_ids: List[str] = None) -> None:
        model_file = os.path.join(model_dir, "openvino_model.xml")
        self.request = ov.Core().compile_model(
            model_file, device_name=device).create_infer_request()
        self.eos_token_ids = eos_token_ids if eos_token_ids is not None else []

    def softmax(self, logits: np.array, temperature: float):
        norm = logits - np.max(logits, axis=-1)
        return np.exp(norm / temperature) / np.sum(np.exp(norm / temperature),
                                                   axis=1)

    def sample_next_token(self,
                          logits: np.array,
                          temperature: float = 0.6,
                          top_p: float = 0.9):
        probs = self.softmax(logits[:, -1], temperature)
        sorted_idx = np.argsort(probs)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
        probs_sum = np.cumsum(sorted_probs, axis=-1)
        mask = probs_sum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= np.sum(sorted_probs, axis=-1, keepdims=True)

        next_token = np.array([
            np.random.choice(sorted_probs.shape[1], p=sorted_probs[i])
            for i in range(sorted_probs.shape[0])
        ])
        sampled_indices = np.take_along_axis(sorted_idx,
                                             next_token[:, np.newaxis],
                                             axis=-1).flatten()
        return sampled_indices

    def generate(self,
                 input_ids: str,
                 max_tokens: int = 500,
                 socketobj: SocketIO = None) -> str:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = np.arange(0, seq_len, dtype=np.int64)
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

            logits_tensor = self.request.get_tensor("logits")
            logits = logits_tensor.data
            next_token = self.sample_next_token(logits)[0].item()

            if next_token in self.eos_token_ids or len(
                    output_tokens) > max_tokens:
                break

            output_tokens += [next_token]
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            new_position_id += 1
            inputs["position_ids"] = new_position_id
            input_ids = np.array([[next_token]], dtype=np.longlong)

            if socketobj:
                latency = time.perf_counter() - start
                socketobj.emit('tps_measure', len(output_tokens) / latency)

        return output_tokens
