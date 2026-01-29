import requests
import json

url = "https://mcp.api-inference.modelscope.net/89672e382eec45/sse"
print(f"Connecting to {url}...")
try:
    response = requests.get(url, stream=True, timeout=10)
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            print(f"Received: {line_str}")
            if line_str.startswith('event: endpoint'):
                continue
            if line_str.startswith('data: '):
                endpoint = line_str[6:]
                print(f"Post endpoint: {endpoint}")
                break
except Exception as e:
    print(f"Error: {e}")
