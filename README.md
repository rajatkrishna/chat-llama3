# Llama 3 Chat

Running Llama 3 inference locally with OpenVINO.

Use the export script `export.py` to quantize and save the model in the OpenVINO IR format.

```
python3 export.py --model-id=meta-llama/Meta-Llama-3-8B-Instruct --export_path "models/llama-3-8b-instruct"
```

Start using `chat.py` script.

```
python3 chat.py --model-path=models/llama-3-8b-instruct
```
