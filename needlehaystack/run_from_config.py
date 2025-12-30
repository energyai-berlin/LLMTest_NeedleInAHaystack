import json
import sys
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from .run import CommandArgs, get_model_to_test, get_evaluator

load_dotenv()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path is None:
        # Default to models_config.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "models_config.json"

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        return json.load(f)


def create_command_args(model_config: Dict, default_params: Dict) -> CommandArgs:
    """Create a CommandArgs object from model config and defaults.

    Args:
        model_config: Model-specific configuration from JSON
        default_params: Default parameters that apply to all models

    Returns:
        CommandArgs object ready to be used with get_model_to_test and get_evaluator
    """
    # Merge default params with model-specific params
    # Skip "enabled" field as it's only used for filtering which models to run
    merged = default_params.copy()
    for key, value in model_config.items():
        if key not in ["enabled"]:
            merged[key] = value

    # Create CommandArgs with all parameters
    return CommandArgs(
        provider=merged.get("provider"),
        evaluator=merged.get("evaluator", "local"),
        model_name=merged.get("model_name"),
        evaluator_model_name=merged.get("evaluator_model_name", "qwen3:0.6b"),
        needle=merged.get(
            "needle",
            "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        ),
        haystack_dir=merged.get("haystack_dir", "PaulGrahamEssays"),
        retrieval_question=merged.get(
            "retrieval_question", "What is the best thing to do in San Francisco?"
        ),
        results_version=merged.get("results_version", 1),
        context_lengths_min=merged.get("context_lengths_min", 9000),
        context_lengths_max=merged.get("context_lengths_max", 25000),
        context_lengths_num_intervals=merged.get("context_lengths_num_intervals", 4),
        context_lengths=merged.get("context_lengths"),
        document_depth_percent_min=merged.get("document_depth_percent_min", 0),
        document_depth_percent_max=merged.get("document_depth_percent_max", 100),
        document_depth_percent_intervals=merged.get(
            "document_depth_percent_intervals", 4
        ),
        document_depth_percents=merged.get("document_depth_percents"),
        document_depth_percent_interval_type=merged.get(
            "document_depth_percent_interval_type", "linear"
        ),
        num_concurrent_requests=merged.get("num_concurrent_requests", 1),
        save_results=merged.get("save_results", True),
        save_contexts=merged.get("save_contexts", True),
        final_context_length_buffer=merged.get("final_context_length_buffer", 200),
        seconds_to_sleep_between_completions=merged.get(
            "seconds_to_sleep_between_completions"
        ),
        print_ongoing_status=merged.get("print_ongoing_status", True),
        device=merged.get("device"),
        multi_needle=merged.get("multi_needle", False),
        needles=merged.get(
            "needles",
            [
                " Figs are one of the secret ingredients needed to build the perfect pizza. ",
                " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ",
                " Goat cheese is one of the secret ingredients needed to build the perfect pizza. ",
            ],
        ),
        eval_set=merged.get("eval_set", "multi-needle-eval-pizza-3"),
        multi_needle_type=merged.get("multi_needle_type", "depth_percent"),
        results_subdir=merged.get("results_subdir"),
    )


def run_model_test(
    model_config: Dict, default_params: Dict, test_num: int, total_tests: int
) -> bool:
    """Run the needle haystack test for a single model configuration."""
    provider = model_config["provider"]
    model_name = model_config["model_name"]

    print("\n" + "-" * 80)
    print(f"Test {test_num}/{total_tests}: {provider} - {model_name}")
    print("-" * 80 + "\n")

    try:
        # Create CommandArgs from config
        args = create_command_args(model_config, default_params)

        # Use existing functions from run.py to get model and evaluator
        args.model_to_test = get_model_to_test(args)
        args.evaluator = get_evaluator(args)

        # Create and run the tester (same logic as run.py)
        if args.multi_needle:
            print("Testing multi-needle")
            tester = LLMMultiNeedleHaystackTester(**args.__dict__)
        else:
            print("Testing single-needle")
            tester = LLMNeedleHaystackTester(**args.__dict__)

        tester.start_test()

        print(f"\n✓ Test completed successfully for {model_name}")
        return True

    except Exception as e:
        print(f"\n✗ Test failed for {model_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run all model tests from configuration."""
    # Get config file path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    default_params = config.get("default_params", {})
    models = config.get("models", [])

    # Filter enabled models - only run tests for models with "enabled": true
    # Models without an "enabled" field default to enabled (True)
    enabled_models = [m for m in models if m.get("enabled", True)]

    if not enabled_models:
        print("Error: No enabled models found in configuration.")
        print(
            f"Please edit {config_path} and set 'enabled': true for at least one model."
        )
        sys.exit(1)

    print("Multi-Model Needle Haystack Testing")
    print(f"Configuration file: {config_path}")
    print("-" * 80)
    print(f"\nTesting {len(enabled_models)} enabled model(s):")
    for i, model in enumerate(enabled_models, 1):
        print(f"  {i}. {model['provider']} - {model['model_name']}")
    print()

    # Run tests for all enabled models
    results = []
    for i, model_config in enumerate(enabled_models, 1):
        success = run_model_test(model_config, default_params, i, len(enabled_models))
        results.append((model_config["model_name"], success))

    # Print summary
    print("\n" + "-" * 80)
    print("Testing Summary")
    print("-" * 80 + "\n")

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    for model_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {model_name}")

    print(f"\nTotal: {len(results)} | Passed: {successful} | Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
