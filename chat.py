from generate import Llama
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        help="Path to the exported model.")
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help="Print time to generate")
    args = parser.parse_args()

    model_path = args.model_path

    model = Llama(model_path, role="Assistant")
    messages = [
        {"role": "system", "content": "You are a helpful AI dialog assistant. AI is helpful, kind, obedient, honest, and knows its own limits."}
    ]
    print("Type 'exit' to stop.\n")
    while True:
        prompt = input("\nUser: ").strip()
        if prompt:
            if prompt.lower() == "exit":
                exit(0)
            messages.append(dict(role="user", content=prompt))
            prompt = model.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            response, ttf, tt = model.generate(prompt, verbose=args.verbose)
            messages.append(dict(role="assistant", content=response))
