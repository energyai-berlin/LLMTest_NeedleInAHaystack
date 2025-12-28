# Needle In A Haystack - LLM Context Testing

Test the in-context retrieval ability of long context LLMs by hiding a "needle" (random fact) in a "haystack" (long context) and measuring if the model can retrieve it.

## Quick Start

```bash
# Install dependencies
uv sync

# Update .env
HF_TOKEN=your_api_key

# Run test
uv run python -m needlehaystack.run --provider huggingface --evaluator local --model_name "Qwen/Qwen3-0.6B"
```

## How It Works

1. Insert a random fact at various depths in a long context
2. Ask the model to retrieve the fact
3. Test across different context lengths and document depths
4. Generate a heatmap of retrieval accuracy

## Supported Providers

- OpenAI (`--provider openai`)
- Anthropic (`--provider anthropic`)
- HuggingFace (`--provider huggingface`)
- Ollama (`--provider ollama`)

## Key Parameters

```bash
--context_lengths "[2000, 4000, 8000]"           # Context sizes to test
--document_depth_percents "[25, 50, 75]"         # Where to place needle (%)
--multi_needle True                               # Test multiple needles
--evaluator openai                                # Evaluation method
```

## Example Tests

```bash
# Single test
needlehaystack.run_test --provider openai \
  --model_name "gpt-4" \
  --context_lengths "[2000]" \
  --document_depth_percents "[50]"

# Full sweep
needlehaystack.run_test --provider anthropic \
  --model_name "claude-3-opus" \
  --context_lengths_min 1000 \
  --context_lengths_max 100000 \
  --context_lengths_num_intervals 10
```

## Results

Results are saved to `results/` directory. Use the included Jupyter notebook to visualize performance heatmaps.

## License

MIT License - see [LICENSE.txt](LICENSE.txt)
