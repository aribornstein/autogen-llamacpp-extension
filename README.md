
# AutoGen Llama-CPP Chat Completion Extension

This extension provides a **Chat Completion Client** using the **Llama-CPP** model. It integrates with the AutoGen ecosystem, enabling AI-powered chat completion with access to external tools.

## Installation

To install the `autogen-llama-cpp-chat-completion` extension, run the following command:

```bash
pip install autogen-llama-cpp-chat-completion
```

### Dependencies
- `autogen-core>=0.4,<0.5`
- `pydantic`
- `llama-cpp`

## Usage

Once installed, you can integrate this extension into your AutoGen system for chat-based completions using the Llama-CPP model.


### Example Usage

Here’s an example of how you can use the extension to create a chat session with Llama-CPP:

```python
from autogen_llama_cpp_chat_completion.llama_cpp_extension import LlamaCppChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

# Initialize the LlamaCpp client
client = LlamaCppChatCompletionClient(
    repo_id="your_repo_id", 
    filename="path_to_model_file", 
    n_gpu_layers=-1, 
    seed=1337, 
    n_ctx=1000, 
    verbose=True
)

# Create chat messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="What is the capital of France?")
]

# Get a response from the model
result = await client.create(messages)

# Print the result
print(result.content)
```

### Phi-4 Model Example

You can also use the `phi-4` model for chat completions. Here's an example of how to integrate it with the extension:

```python
from autogen_llama_cpp_chat_completion.llama_cpp_extension import LlamaCppChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

# Initialize the Phi-4 LlamaCpp client
model_client = LlamaCppChatCompletionClient(
    repo_id="unsloth/phi-4-GGUF",
    filename="phi-4-Q2_K_L.gguf",
    n_gpu_layers=-1,
    seed=1337,
    n_ctx=16384,
    verbose=False,
)

# Create chat messages
messages = [
    SystemMessage(content="You are an assistant with the Phi-4 model."),
    UserMessage(content="What is the latest breakthrough in AI research?")
]

# Get a response from the model
result = await model_client.create(messages)

# Print the result
print(result.content)
```

This example demonstrates how to use the `phi-4` model, providing a larger context window (`n_ctx=16384`) and using the model from the `"unsloth/phi-4-GGUF"` repository.

### Streaming Mode

You can also use streaming mode to generate responses incrementally:

```python
response_generator = client.create_stream(messages)

# Iterate through the response stream
async for token in response_generator:
    print(token)
```

## Configuration

When initializing the `LlamaCppChatCompletionClient`, you can provide the following configuration parameters:

- `repo_id`: The repository ID for the model you want to use.
- `filename`: The path to the model file.
- `n_gpu_layers`: The number of GPU layers (default is -1).
- `seed`: The random seed to use for initialization (default is 1337).
- `n_ctx`: The context window size (default is 1000).
- `verbose`: Whether to print debug information (default is `True`).

## Tools Integration

You can dynamically register tools that the model can use during interaction. If a message invokes a tool, it will be detected, and the corresponding tool will be executed. 

Tools should be passed as part of the `tools` argument when calling the `create` or `create_stream` methods.

### Example Tool Usage

If you have a tool, such as a request validation tool, you can register it and the model will use it if necessary.

```python
tools = [
    Tool(name="validate_request", description="Validates request data")
]

result = await client.create(messages, tools=tools)
```

## Running Tests

To ensure that everything is working correctly, you can run tests with `pytest`:

```bash
pytest
```

## Contributing

If you’d like to contribute to this extension, feel free to open an issue or submit a pull request.

## License

This extension is open source and available under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Topics

- [AutoGen](https://github.com/microsoft/autogen)
- [autogen-extension](https://github.com/topics/autogen-extension)
- [Llama-CPP](https://github.com/facebook/llama-cpp)


