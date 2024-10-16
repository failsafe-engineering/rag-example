# Retrieval Augmented Generation
In the context of large language models, "RAG" stands for Retrieval-Augmented Generation. It refers to a technique that combines the power of a language model (for generation) with external retrieval of information (for augmentation). In a RAG framework, the model retrieves relevant documents or information from an external knowledge base or corpus and uses that data to improve or inform its text generation process.

This allows the model to answer questions or generate text that is more accurate and up-to-date, even if that information wasn't part of the model's original training data. It's commonly used in scenarios where up-to-date or specific factual information is important, such as question answering, summarization, or providing contextually rich responses.

This script asks an OpenAI LLM for the list of winners of the Nobel Prize in physics from 2020 to 2024. Since the LLM was trained before the announcement in 2024 it can only tell the winners from 2020 to 2023. However with RAG we can extend the training of the LLM seamlessly.

## Python Version
```shell
>=3.12
```

## Setup Local Env
```shell
python -m venv .venv
```

## Install
```shell
pip install -r requirements.txt
```

## Setup OpenAI API Key
The script talks to the OpenAI API. A project and API key are needed.
```shell
export OPENAI_API_KEY=<your-api-key>
```

## Run Script without RAG
```shell
python main.py
```

## Run Script with RAG
```shell
python main.py --rag true
```

