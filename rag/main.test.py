import os
import unittest
from unittest import mock
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from main import fetch_context
from main import build_prompt
from main import initialise_llm
from main import build_and_invoke_chain
from main import get_api_key
from main import main


class TestFetchContext(unittest.TestCase):
    def test_fetch_context_success(self):
        mock_url = "https://example.com"
        mock_html = '<html><body><article class="page-content border-top entry-content">Test article content with  extra  spaces</article></body></html>'

        with mock.patch("requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.text = mock_html
            mock_get.return_value = mock_response

            result = fetch_context(mock_url)

            self.assertEqual(result, "Test article content with extra spaces")
            mock_get.assert_called_once_with(mock_url)

    def test_fetch_context_no_article(self):
        mock_url = "https://example.com"
        mock_html = "<html><body><div>No article here</div></body></html>"

        with mock.patch("requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.text = mock_html
            mock_get.return_value = mock_response

            with self.assertRaises(ValueError) as context:
                fetch_context(mock_url)

            self.assertEqual(
                str(context.exception), "Could not find the article content on the page"
            )
            mock_get.assert_called_once_with(mock_url)

    def test_build_prompt(self):
        prompt = build_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
        self.assertEqual(
            prompt.messages[0].prompt.template,
            "\n        You are a helpful assistant. You will be given a context. You must answer the question with the provided context.\n        Context: {context}\n        Question: {question}\n        ",
        )

    def test_initialise_llm(self):
        model = "gpt-4"
        api_key = SecretStr("test_api_key")

        with mock.patch("main.ChatOpenAI") as mock_chat_openai:
            mock_instance = mock.Mock()
            mock_chat_openai.return_value = mock_instance

            result = initialise_llm(model, api_key)

            mock_chat_openai.assert_called_once_with(
                model=model,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key,
            )

            self.assertEqual(result, mock_instance)

    def test_build_and_invoke_chain_valid_inputs(self):
        mock_prompt = mock.Mock(spec=ChatPromptTemplate)
        mock_llm = mock.Mock(spec=ChatOpenAI)
        mock_context = "Test context"
        mock_question = "Test question"
        mock_chain = mock.Mock()
        mock_chain.invoke.return_value = {"content": "Test response"}

        # Configure the mock_prompt to have an __or__ method
        mock_prompt.__or__ = mock.Mock(return_value=mock_chain)

        result = build_and_invoke_chain(
            mock_prompt, mock_llm, mock_context, mock_question
        )

        mock_prompt.__or__.assert_called_once_with(mock_llm)
        mock_chain.invoke.assert_called_once_with(
            {"context": mock_context, "question": mock_question}
        )
        self.assertEqual(result, {"content": "Test response"})

    def test_get_api_key_from_env(self):
        mock_api_key = "test_api_key"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": mock_api_key}):
            result = get_api_key()
            self.assertIsInstance(result, SecretStr)
            self.assertEqual(result.get_secret_value(), mock_api_key)

    def test_get_api_key_prompt_user(self):
        mock_api_key = "user_entered_api_key"
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "getpass.getpass", return_value=mock_api_key
        ) as mock_getpass:
            result = get_api_key()
            self.assertIsInstance(result, SecretStr)
            self.assertEqual(result.get_secret_value(), mock_api_key)
            mock_getpass.assert_called_once_with("Enter your OpenAI API key: ")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), mock_api_key)

    def test_main_with_short_context(self):
        mock_model = "test-model"
        mock_url = "https://example.com"
        mock_question = "Test question?"
        mock_api_key = SecretStr("test_api_key")
        mock_context = "Short context"
        mock_prompt = ChatPromptTemplate.from_template("Test prompt")
        mock_llm = mock.Mock(spec=ChatOpenAI)
        mock_result = mock.Mock()
        mock_result.content = "Test response"

        with mock.patch(
            "main.get_api_key", return_value=mock_api_key
        ) as mock_get_api_key, mock.patch(
            "main.fetch_context", return_value=mock_context
        ) as mock_fetch_context, mock.patch(
            "main.build_prompt", return_value=mock_prompt
        ) as mock_build_prompt, mock.patch(
            "main.initialise_llm", return_value=mock_llm
        ) as mock_initialise_llm, mock.patch(
            "main.build_and_invoke_chain", return_value=mock_result
        ) as mock_build_and_invoke_chain:
            result = main(mock_model, mock_url, mock_question, True)

            mock_get_api_key.assert_called_once()
            mock_fetch_context.assert_called_once_with(mock_url)
            mock_build_prompt.assert_called_once()
            mock_initialise_llm.assert_called_once_with(mock_model, mock_api_key)
            mock_build_and_invoke_chain.assert_called_once_with(
                mock_prompt, mock_llm, mock_context, mock_question
            )
            self.assertEqual(result, "Test response")


if __name__ == "__main__":
    unittest.main()
