import getpass
import os
import argparse
import requests
from bs4 import BeautifulSoup
from typing import Optional, Any
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

OPENAI_MODEL: str = "gpt-4o-mini-2024-07-18"
AUGMENTATION_URL: str = "https://www.nobelprize.org/prizes/physics/2024/press-release/"
QUESTION: str = (
    "Please give me a list with the name Nobel Prize winners in physics from 2020 to 2024 and a maximum one sentence description of their contribution. If there are multiple Nobel Prize winners in the same year, mentioned them in the same sentence."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG example script")
    parser.add_argument(
        "--rag",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Enable or disable RAG (default: false)",
    )
    return parser.parse_args()


def fetch_context(url: str) -> str:
    """
    Fetches and processes the content of a web page, extracting the main article text.

    This function sends a GET request to the specified URL, parses the HTML content,
    and extracts the text from the main article element. It then cleans and
    normalizes the text by removing extra whitespace.

    Parameters:
    url (str): The URL of the web page to fetch and process.

    Returns:
    str: A single string containing the cleaned and normalized text content of the main article.

    Raises:
    ValueError: If the article content cannot be found on the page.
    """
    response: requests.Response = requests.get(url)
    soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article", {"class": "page-content border-top entry-content"})
    if article is None:
        raise ValueError("Could not find the article content on the page")
    text_content = article.text.strip()
    return " ".join(text_content.split())


def build_prompt() -> ChatPromptTemplate:
    """
    Builds and returns a ChatPromptTemplate for the conversation with the AI assistant.

    This function creates a template for the conversation, instructing the AI to act as a
    helpful assistant that answers questions based on the provided context.

    Returns:
    ChatPromptTemplate: A template object containing the structure for the conversation,
                        including placeholders for the context and question.
    """
    return ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. You will be given a context. You must answer the question with the provided context.
        Context: {context}
        Question: {question}
        """
    )


def initialise_llm(model: str, api_key: SecretStr) -> ChatOpenAI:
    """
    Initialize and configure a ChatOpenAI instance with specified parameters.

    This function creates a new ChatOpenAI object with predefined settings for
    temperature, max_tokens, timeout, and max_retries. It uses the provided
    model name and API key to set up the instance.

    Parameters:
    model (str): The name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo").
    api_key (SecretStr): The OpenAI API key as a SecretStr object for secure handling.

    Returns:
    ChatOpenAI: A configured instance of the ChatOpenAI class ready for use in
                language model interactions.
    """
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )


def build_and_invoke_chain(
    prompt: ChatPromptTemplate, llm: ChatOpenAI, context: str, question: str
) -> Any:
    """
    Builds and invokes a language model chain using the provided prompt, LLM, context, and question.

    This function creates a chain by combining the prompt template with the language model,
    then invokes the chain with the given context and question.

    Parameters:
    prompt (ChatPromptTemplate): The chat prompt template to be used in the chain.
    llm (ChatOpenAI): The language model instance to be used for generating responses.
    context (str): The context information to be provided to the language model.
    question (str): The question to be answered by the language model.

    Returns:
    Any: The result of invoking the chain, typically containing the model's response.
    """
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question})


def get_api_key() -> SecretStr:
    """
    Retrieves the OpenAI API key from environment variables or prompts the user for input.

    This function first checks if the API key is available in the environment variables.
    If not found, it prompts the user to enter the key interactively. The entered key
    is then stored in the environment variables for future use.

    Returns:
    SecretStr: A SecretStr object containing the OpenAI API key, providing secure
               handling of the sensitive information.
    """
    api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return SecretStr(api_key)


def main(
    model: str = OPENAI_MODEL,
    url: str = AUGMENTATION_URL,
    question: str = QUESTION,
    use_rag: bool = False,
) -> str:
    """
    Orchestrates the process of fetching context, building a prompt, initializing a language model,
    and invoking a chain to generate a response to a given question.

    This function serves as the main entry point for the question-answering system. It retrieves
    the API key, fetches context from a URL, builds a prompt, initializes the language model,
    and then invokes a chain to generate a response based on the context and question.

    Parameters:
    model (str): The name of the language model to use. Defaults to LLM_MODEL.
    url (str): The URL from which to fetch the context. Defaults to AUGMENTATION_URL.
    question (str): The question to be answered. Defaults to QUESTION.

    Returns:
    str: The content of the generated response from the language model.
    """
    api_key: SecretStr = get_api_key()
    context: str = fetch_context(url) if use_rag else ""
    prompt: ChatPromptTemplate = build_prompt()
    llm: ChatOpenAI = initialise_llm(model, api_key)
    response: Any = build_and_invoke_chain(prompt, llm, context, question)
    return vars(response)["content"]


if __name__ == "__main__":
    args = parse_arguments()
    use_rag = args.rag.lower() == "true"
    print(f"Question:\n{QUESTION}", end="\n\n")
    result = main(use_rag=use_rag)
    result = result.strip().replace("\n\n", "\n")
    print(f"Answer:\n{result}", end="\n\n")
