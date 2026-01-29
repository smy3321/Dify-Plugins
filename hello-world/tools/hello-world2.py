from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import requests

class HelloWorldTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        url = tool_parameters.get("url", "")
        
        try:
            # 发送请求
            response = requests.get(url, timeout=10)
            text=response.text[:1000]
            
            
            yield self.create_json_message({"text": text,"parameters": tool_parameters})
            
        except requests.RequestException as e:
            yield self.create_text_message(f"请求失败: {str(e)}")
        except Exception as e:
            yield self.create_text_message(f"处理失败: {str(e)}")