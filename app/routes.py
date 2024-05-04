from flask import render_template, session
from app import app, socketio
from app.assistant import LlamaAssistant
import os
from app.prompts import SYS_PROMPT


def create_chat(model_dir: str, device: str = "CPU"):
    return LlamaAssistant(model_dir=model_dir, device=device, sys_prompt=SYS_PROMPT)


MODEL_DIR = os.environ.get(
    "MODEL_DIR", "models/llama-3-instruct-8b")
DEVICE = os.environ.get("DEVICE", "CPU")
assistant = create_chat(MODEL_DIR, device=DEVICE)


@app.route('/')
def sessions():
    return render_template('chat.html')


@socketio.on('msg_receive')
def handle_msg_receive(query: dict):
    message = query['message']

    if 'msgs' not in session:
        session['msgs'] = []

    session['msgs'].append(dict(role="user", content=message))
    response = assistant.chat(msgs=session.get('msgs'), socketobj=socketio)
    session['msgs'].append(dict(role="assistant", content=response))
    socketio.emit('msg_response', dict(answer=response))
