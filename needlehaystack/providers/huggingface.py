from operator import itemgetter
from typing import Optional, Union
import torch
import os
from pathlib import Path


# Answer is I love TOlba score 0 100


import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .model import ModelProvider

SYSTEM_PROMPT = """You are a precise information retrieval assistant. Your task is to find and extract the exact answer from the provided context.

CRITICAL RULES:
1. ONLY provide the direct answer from the context
2. If the answer is in the context, state it exactly as written
3. Do NOT make up information or guess
4. Do NOT add explanations or extra commentary"""

CONTEXT_PROMPT = """
<context>
{context}
</context>
"""

QUESTION_PROMPT = """
<question>
{question}
</question>
"""


class HuggingFace(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """

    DEFAULT_MODEL_KWARGS: dict = dict(max_new_tokens=300)

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        device: Optional[Union[int, str]] = None,
    ):
        """
        Initializes the HuggingFace model provider with a specific model.

        Args:
            model_name (str): The path of the HuggingFace model to use. Can be either:
                - A HuggingFace Hub model ID (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
                - A local directory path containing the model files
                Defaults to 'mistralai/Mistral-7B-Instruct-v0.2'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
            device (Optional[Union[int, str]]): Device to run the model on.
                - 0 or positive int: Single GPU device ID
                - -1: CPU
                - "auto": Automatic multi-GPU distribution (recommended for large models)
                - "balanced": Balanced multi-GPU distribution
                - "balanced_low_0": Balanced distribution with less weight on GPU 0
                - "sequential": Sequential layer distribution across GPUs
                - None (default): Auto-detect (GPU 0 if available, otherwise CPU)

        Raises:
            ValueError: If HF_TOKEN is not found in the environment and loading from Hub.
        """

        self.model_name = model_name
        self.model_kwargs = model_kwargs

        # Check if model_name is a local path
        is_local_path = Path(model_name).exists() and Path(model_name).is_dir()

        # Only require HF_TOKEN for remote models
        if not is_local_path:
            HF_TOKEN = os.getenv("HF_TOKEN")
            if not HF_TOKEN:
                raise ValueError(
                    "HF_TOKEN must be in env for loading models from HuggingFace Hub."
                )
            self.api_key = HF_TOKEN
            huggingface_hub.login(self.api_key)

        # Determine device strategy
        use_device_map = isinstance(device, str)

        if use_device_map:
            # Multi-GPU mode with device_map
            device_map = device
            pipeline_device = None  # Pipeline will use device_map from model
            torch_dtype = torch.float16
            print(f"Using device_map='{device_map}' for multi-GPU distribution")
        else:
            # Single device mode
            device_map = None
            if device is None:
                pipeline_device = 0 if torch.cuda.is_available() else -1
            else:
                pipeline_device = device
            torch_dtype = torch.float16 if pipeline_device >= 0 else torch.float32
            print(f"Using single device: {pipeline_device}")

        # Load model based on whether it's local or remote
        if is_local_path:
            print(f"Loading model from local directory: {model_name}")
            # Load tokenizer from local directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )

            # Load model from local directory
            local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=device_map,  # Will be None for single GPU, or "auto"/"balanced" etc. for multi-GPU
            )

            # Create pipeline with local model
            pipe = pipeline(
                "text-generation",
                model=local_model,
                tokenizer=self.tokenizer,
                device=pipeline_device,  # Will be None for multi-GPU mode
                **model_kwargs,
            )

            self.model = HuggingFacePipeline(pipeline=pipe)
        else:
            print(f"Loading model from HuggingFace Hub: {model_name}")
            # Load from HuggingFace Hub
            if use_device_map:
                # For multi-GPU with Hub models, we need to load manually
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                hub_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )

                pipe = pipeline(
                    "text-generation",
                    model=hub_model,
                    tokenizer=self.tokenizer,
                    device=None,
                    **model_kwargs,
                )

                self.model = HuggingFacePipeline(pipeline=pipe)
            else:
                # Single device mode (original behavior)
                self.model = HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    device=pipeline_device,
                    task="text-generation",
                    pipeline_kwargs=model_kwargs,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    async def evaluate_model(self, prompt_template: str) -> str:
        """
        Evaluates a given prompt using the Ollama model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt, with thinking tags removed.
        """
        prompt = PromptTemplate.from_template(prompt_template)

        chain = LLMChain(llm=self.model, prompt=prompt)

        response = await chain.ainvoke(input={})

        # Extract the text from the response dictionary
        if isinstance(response, dict):
            response_text = response.get("text", "")
        else:
            response_text = str(response)

        return response_text

    def generate_prompt(
        self, context: str, retrieval_question: str
    ) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        prompt_format = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(context=context)
                + "\n\n"
                + QUESTION_PROMPT.format(question=retrieval_question)
                + "\n\nProvide ONLY the exact answer from the context.",
            },
        ]

        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        if (
            "system" not in self.tokenizer.chat_template
            or "raise_exception('System role not supported')"
            in self.tokenizer.chat_template
        ):
            prompt_format.pop(0)
            prompt_format[0]["content"] = (
                SYSTEM_PROMPT + "\n" + prompt_format[0]["content"]
            )

        return self.tokenizer.apply_chat_template(
            prompt_format,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode_text_to_tokens(self, text: str, no_bos: bool = False) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.
            no_bos (bool): If True, the tokenization result will not contain bos token at first.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        if no_bos:
            return self.tokenizer.encode(text)[1:]
        else:
            return self.tokenizer.encode(text)

    def decode_tokens(
        self, tokens: list[int], context_length: Optional[int] = None
    ) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])

    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question,
        queries the HuggingFace model, and returns the model's response. This method leverages the LangChain
        library to build a sequence of operations: extracting input variables, generating a prompt,
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question.
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a
            dynamically provided question. The runnable encapsulates the entire process from prompt
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """
        You are a precise information retrieval assistant. Find and extract the exact answer from the context.

        CRITICAL RULES:
        - ONLY provide the direct answer from the context
        - Do NOT make up information or guess

        Context:
        {context}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        # Use the already loaded model instead of creating a new one
        chat_model = ChatHuggingFace(llm=self.model)
        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}
            | prompt
            | chat_model
        )
        return chain
