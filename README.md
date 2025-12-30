# Needle In A Haystack - LLM Context Testing

Test the in-context retrieval ability of long context LLMs by hiding a "needle" (random fact) in a "haystack" (long context) and measuring if the model can retrieve it.

## Quick Start

```bash
# Install dependencies
uv sync

# Update .env (only needed for HuggingFace Hub models)
HF_TOKEN=your_api_key

# Run test with HuggingFace Hub model
uv run python -m needlehaystack.run --provider huggingface --evaluator local --model_name "Qwen/Qwen3-0.6B"

# Run test with local model
uv run python -m needlehaystack.run --provider huggingface --evaluator local --model_name "/path/to/local/model"

# Run test with specific GPU device
uv run python -m needlehaystack.run --provider huggingface --evaluator local --model_name "Qwen/Qwen3-0.6B" --device 0

# Run test on CPU
uv run python -m needlehaystack.run --provider huggingface --evaluator local --model_name "Qwen/Qwen3-0.6B" --device -1
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
  - Supports both HuggingFace Hub models and local model directories
  - For local models, provide the absolute path to the model directory
- Ollama (`--provider ollama`)

## Key Parameters

```bash
--context_lengths "[2000, 4000, 8000]"           # Context sizes to test
--document_depth_percents "[25, 50, 75]"         # Where to place needle (%)
--multi_needle True                               # Test multiple needles
--evaluator openai                                # Evaluation method (openai or local)
--device 0                                        # GPU device (0, 1, etc.) or -1 for CPU (HuggingFace only)
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

Results are automatically organized and visualized:

- Each test run creates a timestamped subdirectory in `results/` (e.g., `results/run_20241228_153045/`)
- Individual test results are saved as JSON files in the run subdirectory
- A visualization heatmap (`visualization.png`) is automatically generated after each test
- The heatmap shows model performance across different context lengths and document depths

### Automatic Visualization

After each test completes, a heatmap visualization is automatically generated and saved to the results subdirectory. The visualization shows:
- **X-axis**: Context lengths (tokens)
- **Y-axis**: Document depth percentages
- **Colors**: Retrieval scores (red = poor, yellow = medium, green = excellent)

### Manual Visualization

You can also generate visualizations manually:

```python
from needlehaystack import generate_visualization

generate_visualization(
    results_dir="results/run_20241228_153045",
    model_name="My Model",
    output_path="custom_viz.png"
)
```

## Multi-Model Testing

Test multiple models automatically using a JSON configuration file.

### Quick Start

1. Edit `needlehaystack/models_config.json` to configure your models:

```json
{
  "default_params": {
    "context_lengths_min": 9000,
    "context_lengths_max": 25000,
    "device": 0,
    ...
  },
  "models": [
    {
      "provider": "huggingface",
      "model_name": "meta-llama/Llama-3.3-70B-Instruct",
      "evaluator": "local",
      "evaluator_model_name": "qwen3:0.6b",
      "enabled": true
    },
    {
      "provider": "openai",
      "model_name": "gpt-4-turbo-preview",
      "enabled": false
    }
  ]
}
```

2. Run the script:

```bash
uv run python -m needlehaystack.run_from_config
```

### Configuration Parameters

**Model-Specific Parameters:**
- `provider`: Model provider - `"openai"`, `"anthropic"`, or `"huggingface"`
- `model_name`: Full model name/identifier
- `evaluator`: Evaluator to use - `"openai"` or `"local"` (Ollama)
  - Local evaluator scores: 0 (unrelated), 25 (minor relevance), 50 (moderate), 75 (aligned with minor omissions), 100 (perfect)
- `evaluator_model_name`: Model name for the evaluator (default: `"qwen3:0.6b"`)
- `device`: GPU device ID (0, 1, etc.) or -1 for CPU (HuggingFace only, optional)

**Test Parameters (typically in `default_params`):**
- `context_lengths_min`: Minimum context length to test (default: 9000)
- `context_lengths_max`: Maximum context length to test (default: 25000)
- `context_lengths_num_intervals`: Number of intervals between min and max (default: 4)
- `document_depth_percent_intervals`: Number of depth positions to test (default: 4)
- `num_concurrent_requests`: Number of concurrent API requests (default: 1)
- `save_results`: Save test results to JSON files (default: true)
- `save_contexts`: Save context files (default: true)
- `device`: GPU device ID for HuggingFace models (default: auto-detect, can be overridden per model)
- `multi_needle`: Enable multi-needle testing (default: false)
- `results_subdir`: Custom subdirectory for results (optional)

**Default Test Content (in `default_params`):**
- `needle`: The text to hide in the haystack
- `retrieval_question`: Question to ask the model
- `haystack_dir`: Directory containing background text files

### Examples

**Testing Multiple HuggingFace Models on Different GPUs:**

```json
{
  "default_params": {
    "device": 0
  },
  "models": [
    {
      "provider": "huggingface",
      "model_name": "meta-llama/Llama-3.3-70B-Instruct",
      "enabled": true
    },
    {
      "provider": "huggingface",
      "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
      "device": 1,
      "enabled": true,
      "comment": "Override device to use GPU 1 for this model"
    }
  ]
}
```

**Testing with Different Context Lengths:**

```json
{
  "models": [
    {
      "provider": "openai",
      "model_name": "gpt-4-turbo-preview",
      "context_lengths_min": 10000,
      "context_lengths_max": 100000,
      "context_lengths_num_intervals": 10,
      "enabled": true
    }
  ]
}
```

**Multi-Needle Testing:**

```json
{
  "default_params": {
    "device": 0,
    "multi_needle": true,
    "needles": [
      " Figs are one of the secret ingredients. ",
      " Prosciutto is one of the secret ingredients. ",
      " Goat cheese is one of the secret ingredients. "
    ]
  },
  "models": [...]
}
```

### Output

Results are saved to:
- `results/run_YYYYMMDD_HHMMSS/` - Test results as JSON files
- `contexts/run_YYYYMMDD_HHMMSS/` - Context files used in tests
- `results/run_YYYYMMDD_HHMMSS/visualization.png` - Visualization of results

Each model test creates its own timestamped subdirectory unless you specify `results_subdir`.

### Tips

1. **Enable/Disable Models**: Set `"enabled": false` to skip a model without deleting its configuration
2. **Rate Limits**: Adjust `num_concurrent_requests` and add `seconds_to_sleep_between_completions` to avoid rate limits
3. **Quick Tests**: Use smaller `context_lengths_num_intervals` and `document_depth_percent_intervals` for faster testing
4. **Cost Control**: Test expensive models with smaller intervals first before full runs
5. **Local Evaluator**: Use `"evaluator": "local"` with Ollama to avoid API costs for evaluation

### Troubleshooting

- **API Keys**: Ensure your `.env` file or environment has the necessary API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **Ollama**: If using local evaluator, ensure Ollama is running and has the evaluator model installed
- **Memory**: Large context lengths may require significant RAM, especially for local models
- **Failed Tests**: The script continues even if one model fails; check the summary at the end

## License

MIT License - see [LICENSE.txt](LICENSE.txt)
