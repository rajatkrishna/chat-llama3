FROM python:3.11

WORKDIR /chat-app
ADD llama ./llama
ADD llama_assistant ./llama_assistant
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r ./requirements.txt

ENV FLASK_APP "llama_assistant"
ENV MODEL_DIR "models/llama-3.1-instruct-8b"
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
