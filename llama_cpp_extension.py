from autogen_core import CancellationToken
from pydantic import BaseModel
from autogen_core.models import ChatCompletionClient, CreateResult, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tools import Tool
from llama_cpp import Llama
from typing import List, Dict, Any, Optional
import json

class LlamaCppChatCompletionClient(ChatCompletionClient):
    def __init__(self, repo_id: str, filename: str, n_gpu_layers: int = -1, seed: int = 1337, n_ctx: int = 1000, verbose: bool = True):
        """
        Initialize the LlamaCpp client.
        """
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async def create(self, messages: List[Any], tools: List[Any] = None, **kwargs) -> CreateResult:
        """
        Generate a response using the model, incorporating tool metadata.
        """
        tools = tools or []
        converted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                converted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                converted_messages.append({"role": "assistant", "content": msg.content})
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        # Add tool descriptions to the system message
        tool_descriptions = "\n".join([f"Tool: {i+1}. {tool.name} - {tool.description}" for i, tool in enumerate(tools)])
        system_message = f"You are an assistant with access to tools.\n{tool_descriptions}"
        converted_messages.insert(0, {"role": "system", "content": system_message})

        response = self.llm.create_chat_completion(messages=converted_messages, stream=False)
        self._total_usage["prompt_tokens"] += response.get("usage", {}).get("prompt_tokens", 0)
        self._total_usage["completion_tokens"] += response.get("usage", {}).get("completion_tokens", 0)

        response_text = response["choices"][0]["message"]["content"]

        # Detect tool usage in the response
        tool_call = await self._detect_and_execute_tool(response_text, tools)
        create_result = CreateResult(
            content=tool_call if tool_call else response_text,
            usage=response.get("usage", {}),
            finish_reason=response["choices"][0].get("finish_reason", "unknown"),
            cached=False,
        )
        return create_result

    async def _detect_and_execute_tool(self, response_text: str, tools: List[Tool]) -> Optional[str]:
        """
        Detect if the model is requesting a tool and execute the tool.
        """
        for tool in tools:
            if tool.name.lower() in response_text.lower():  
                func_args = self._extract_tool_arguments(response_text)
                if func_args:
                    args_model = tool.args_type()
                    if "request" in args_model.__fields__:
                        func_args = {"request": func_args}
                    args_instance = args_model(**func_args)
                try:
                    result = await tool.run(args=args_instance, cancellation_token=CancellationToken())
                    return json.dumps(result) if isinstance(result, dict) else str(result)
                except Exception as e:
                    return f"Error executing tool '{tool.name}': {e}"

        return None

    def _extract_tool_arguments(self, response_text: str) -> Dict[str, Any]:
        """
        Extract tool arguments from the response text.
        """
        try:
            args_start = response_text.find("{")
            args_end = response_text.find("}")
            if args_start != -1 and args_end != -1:
                args_str = response_text[args_start:args_end + 1]
                return json.loads(args_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse arguments: {e}")
        return {}

    def actual_usage(self) -> Dict[str, int]:
        return self._total_usage

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {"chat": True}

    def count_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> int:
        return sum(len(msg["content"].split()) for msg in messages)

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "name": "llama-cpp",
            "capabilities": {"chat": True},
            "context_window": self.llm.n_ctx,
        }

    def remaining_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> int:
        used_tokens = self.count_tokens(messages)
        return max(self.llm.n_ctx - used_tokens, 0)

    def total_usage(self) -> Dict[str, int]:
        return self._total_usage
