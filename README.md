# Local Llama 3 Chat 🦙

A simple chat application with LLama 3 using OpenVINO Runtime for inference and transformers library for tokenization.

<p align="center">
    <img width="450" src="https://github.com/rajatkrishna/chat-llama3/assets/61770314/5a7778fc-2de0-4c8c-ab4f-09843c78a2f0">
</p>

<br />

- [Model Export](#model-export)
- [Quickstart with Docker](#quickstart-with-docker)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Export from HuggingFace](#export-from-huggingface)

### Model Export

Download the INT-4 quantized `Meta-Llama-3-8B-Instruct` model already converted to the OpenVINO IR format from [HuggingFace](https://huggingface.co/rajatkrishna/Meta-Llama-3-8B-Instruct-OpenVINO-INT4) using `huggingface-cli` with the following command:

```
huggingface-cli download rajatkrishna/Meta-Llama-3-8B-Instruct-OpenVINO-INT4 --local-dir models/llama-3-instruct-8b
```

### Quickstart with Docker

- [Install docker](https://docs.docker.com/engine/install/).

- Build the docker image with the following command. The source files and model weights are pulled using git, requiring an active internet connection.

    ```
    docker build -t chat-llama .
    ```

- Mount the model directory and start the container using:

    ```
    docker run -v $(pwd)/models:/chat-app/models -p 5000:5000 chat-llama
    ```

    This should start the Flask dev server available on `http://localhost:5000`

### Requirements

- Python 3.11

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

### Export from HuggingFace

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
