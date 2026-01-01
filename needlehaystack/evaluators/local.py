from .evaluator import Evaluator

from langchain_community.chat_models import ChatOllama


class OllamaEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)

    CRITERIA = {
        "accuracy": """
        Score 0: The answer is completely unrelated to the reference.
        Score 25: The answer has minor relevance but does not align with the reference.
        Score 50: The answer has moderate relevance but contains inaccuracies.
        Score 75: The answer aligns with the reference but has minor omissions.
        Score 100: The answer is completely accurate and aligns perfectly with the reference.

        IMPORTANT: You MUST respond with ONLY a valid numerical score (0, 25, 50, 75, or 100).
        Do NOT include any explanations, words, or extra text - return ONLY the number.
        """
    }

    def __init__(
        self,
        model_name: str = "llama3",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        true_answer: str = None,
        question_asked: str = None,
    ):
        """
        :param model_name: Ollama model name (e.g. llama3, mistral, qwen2.5)
        :param model_kwargs: Model configuration (temperature, etc.)
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if not true_answer or not question_asked:
            raise ValueError(
                "true_answer and question_asked must be supplied with init."
            )

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        # Ollama LLM (no API key needed)
        self.evaluator = ChatOllama(
            model=self.model_name,
            **self.model_kwargs,
        )

    def evaluate_response(self, response: str) -> int:
        # Create evaluation prompt
        prompt = f"""
        You are evaluating the accuracy of an answer.

        Question: {self.question_asked}
        Reference Answer: {self.true_answer}
        Given Answer: {response}

        Scoring Criteria:
        {self.CRITERIA['accuracy']}

        Please evaluate the given answer and respond with ONLY a single numerical score (0, 25, 50, 75, or 100).

        CRITICAL: Your response MUST be a valid number from the scoring criteria above. Do not write any words or explanations.
        """

        # Get response from Ollama
        result = self.evaluator.invoke(prompt)

        # Extract score from response
        score_text = result.content.strip()

        # Parse the numerical score
        try:
            score = int(score_text)
        except ValueError:
            # Try to extract first number if response contains extra text
            import re

            numbers = re.findall(r"\d+", score_text)
            if numbers:
                score = int(numbers[0])
            else:
                # Return 0 if score cannot be parsed
                score = 0

        return score
