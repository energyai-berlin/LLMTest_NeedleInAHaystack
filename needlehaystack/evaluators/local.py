from .evaluator import Evaluator

from langchain_community.chat_models import ChatOllama


class OllamaEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)

    CRITERIA = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Only respond with a numerical score
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
            raise ValueError("true_answer and question_asked must be supplied with init.")

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
        prompt = f"""You are evaluating the accuracy of an answer.

Question: {self.question_asked}
Reference Answer: {self.true_answer}
Given Answer: {response}

Scoring Criteria:
{self.CRITERIA['accuracy']}

Please evaluate the given answer and respond with ONLY a single numerical score (1, 3, 5, 7, or 10)."""

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
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
            else:
                raise ValueError(f"Could not parse score from response: {score_text}")

        return score
