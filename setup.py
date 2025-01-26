from setuptools import setup, find_packages

setup(
    name='autogen-llama-cpp-chat-completion',
    version='0.1.0',
    description='Chat completion client extension using Llama CPP',
    packages=find_packages(),
    install_requires=[
        'autogen-core>=0.4,<0.5',
        'pydantic',
        'llama-cpp',
    ],
    entry_points={
        "autogen.extensions": [
            "llama-cpp-chat-completion = autogen_llama_cpp_chat_completion.llama_cpp_extension:LlamaCppChatCompletionClient",
        ],
    },
)
