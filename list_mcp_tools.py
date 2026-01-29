import requests
import json
import time

sse_url = "https://mcp.api-inference.modelscope.net/89672e382eec45/sse"
print(f"Connecting to {sse_url}...")

response = requests.get(sse_url, stream=True, timeout=10)
post_url = None
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            post_url = line_str[6:]
            break

if post_url:
    if not post_url.startswith('http'):
        from urllib.parse import urljoin
        post_url = urljoin(sse_url, post_url)
    
    print(f"Post URL: {post_url}")
    
    # MCP List Tools
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    res = requests.post(post_url, json=payload)
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
else:
    print("Failed to get post_url")
