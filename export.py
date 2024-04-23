from optimum.intel.openvino import OVWeightQuantizationConfig, OVModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str,
                        help="Model id in HuggingFace")
    parser.add_argument('--int8', type=bool, default=False,
                        help="Export in 8-bit format. Defaults to int4")
    parser.add_argument('--ratio', type=float, default=None,
                        help="Ratio of 4-bit vs 8-bit quantized layers")
    parser.add_argument('--sym', type=bool, default=True,
                        help="Symmetric or asymmetric quantization")
    parser.add_argument('--export-path', type=str,
                        help="Directory to export to")
    args = parser.parse_args()

    model_name = args.model_id if args.model_id else 'meta-llama/Meta-Llama-3-8B-Instruct'
    export_path = args.export_path if args.export_path else f"./models/{model_name}"
    start = time.perf_counter()
    if args.int8:
        print("Loading in 8-bit")
        start = time.perf_counter()
        model = OVModelForCausalLM.from_pretrained(
            model_name, export=True, load_in_8bit=True)
    else:
        print("Loading in 4-bit")
        q_config = OVWeightQuantizationConfig(
            bits=4, sym=args.sym, group_size=128, ratio=args.ratio)
        model = OVModelForCausalLM.from_pretrained(
            model_name, export=True, quantization_config=q_config)
    time_to_export_sec = time.perf_counter() - start
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    print(
        f"Model {model_name} saved to {export_path} in {time_to_export_sec:.2f} seconds")
