from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

python_path = "/home/export/doriancl/code/Fall-2025-ML-proj/.venv/bin/python"

server_process, port = launch_server_cmd(
    f"""
{python_path} -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
 --host 0.0.0.0 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")

import openai

client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")
