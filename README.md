# Local Llama Chat 🦙

A simple chat interface to run the Llama 3 model locally using [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino) for inference, transformers library for tokenization and Flask for the chat interface. 

- [Requirements](#requirements)
- [Model Export](#model-export)
- [Getting Started](#getting-started)


### Requirements

- Python 3.11

### Model Export

To download the original model weights from HuggingFace, visit the [HuggingFace model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and accept their License. Once your request has been accepted, use `huggingface-cli` to login to your HuggingFace account in your current runtime with the following command:

```
huggingface-cli login
```

- To download the INT-4 quantized `meta-llama/Meta-Llama-3-8B-Instruct` model already converted to the OpenVINO IR format, you can use the following command:

    ```
    huggingface-cli download rajatkrishna/Meta-Llama-3-8B-Instruct-OpenVINO-INT4 --local-dir models/llama-3-instruct-8b
    ```

- To export the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model quantized to **INT-8** format yourself using [optimum-intel](https://github.com/huggingface/optimum-intel) CLI, install the requirements in `requirements_export.txt`:

    ```
    pip install -r requirements_export.txt
    ```

    Then run the following from the project root:

    ```
    optimum-cli export openvino --model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int8 models/llama-3-instruct-8b
    ```

- Alternately, use the following steps to export the **INT-4** quantized model using the Python API:

    1. Import the dependencies:

        ```
        >>> from optimum.intel.openvino import OVWeightQuantizationConfig, OVModelForCausalLM
        >>> from transformers import AutoTokenizer
        ```

    2. Load the model using `OVModelForCausalLM` class. Set `export=True` to export the model on the fly. 

        ```
        >>> export_path = "models/llama-3-instruct-8b"
        >>> q_config = OVWeightQuantizationConfig(bits=4, sym=True, group_size=128)
        >>> model = OVModelForCausalLM.from_pretrained(model_name, export=True, quantization_config=q_config)
        >>> model.save_pretrained(export_path)
        ```

    3. Now use `AutoTokenizer` to save the tokenizer.

        ```
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> tokenizer.save_pretrained(export_path)
        ```

### Getting Started

1. Clone the repository

    ```
    git clone https://github.com/rajatkrishna/llama3-openvino
    ```

2. Create a new virtual environment to avoid dependency conflicts:

    ```
    python3 -m venv create .env
    source .env/bin/activate
    ```

3. Install the dependencies in `requirements.txt`

    ```
    pip install -r requirements.txt
    ```

4. Start the flask server from the project root using

    ```
    python3 -m flask run
    ```

