from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='autogen-llama-cpp-chat-completion',  # Replace with your package name
    version='0.1.2',  # Replace with your version
    description='A chat completion client extension using Llama CPP, integrating with AutoGen for AI-powered chat interactions.',
    long_description=long_description,  # This is the long description read from README.md
    long_description_content_type='text/markdown',  # Specify the content type of the long description (markdown)
    packages=find_packages(),  # Automatically find all packages in the current directory
    install_requires=[
        'autogen-core>=0.4,<0.5',
        'pydantic',
        'llama-cpp-python>=0.1.0',
    ],
    entry_points={
        "autogen.extensions": [
            "llama-cpp-chat-completion = autogen_llama_cpp_chat_completion.llama_cpp_extension:LlamaCppChatCompletionClient",
        ],
    },
)
