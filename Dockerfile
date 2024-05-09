FROM python:3.11

RUN apt-get update \
    && apt-get install -y git git-lfs \
    && git lfs install

WORKDIR /chat-app
RUN git clone --depth 1 https://huggingface.co/rajatkrishna/Meta-Llama-3-8B-Instruct-OpenVINO-INT4 \
    && git clone --depth 1 https://github.com/rajatkrishna/chat-llama3.git

WORKDIR /chat-app/chat-llama3

RUN mkdir -p models/llama-3-instruct-8b \
    && mv /chat-app/Meta-Llama-3-8B-Instruct-OpenVINO-INT4/* models/llama-3-instruct-8b

RUN pip install --upgrade pip \
    && pip install -r ./requirements.txt

ENV FLASK_APP "app"
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
