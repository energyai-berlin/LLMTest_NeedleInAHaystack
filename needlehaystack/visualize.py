"""
Visualization module for Needle in a Haystack test results.

This module creates heatmap visualizations from test results, showing model performance
across different context lengths and document depths.
"""

import glob
import json
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load test results from JSON files in the specified directory.

    Args:
        results_dir: Path to the directory containing JSON result files

    Returns:
        DataFrame with columns: Document Depth, Context Length, Score
    """
    json_files = glob.glob(f"{results_dir}/*.json")

    if not json_files:
        raise ValueError(f"No JSON files found in {results_dir}")

    data = []

    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)
            # Extract the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)

            data.append(
                {
                    "Document Depth": document_depth,
                    "Context Length": context_length,
                    "Score": score,
                }
            )

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} test results from {results_dir}")

    return df


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table from the results DataFrame.

    This aggregates multiple runs by averaging scores for the same
    context length and document depth combinations.

    Args:
        df: DataFrame with test results

    Returns:
        Pivot table with Document Depth as index, Context Length as columns
    """
    # Aggregate by mean (useful if tests were run multiple times)
    pivot_table = pd.pivot_table(
        df, values="Score", index=["Document Depth", "Context Length"], aggfunc="mean"
    ).reset_index()

    # Convert to proper pivot format
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )

    return pivot_table


def create_heatmap(
    pivot_table: pd.DataFrame,
    model_name: str = "LLM",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (17.5, 8),
    show_plot: bool = False,
) -> None:
    """
    Create and save a heatmap visualization of the test results.

    Args:
        pivot_table: Pivot table with test results
        model_name: Name of the model being tested
        output_path: Path to save the visualization (if None, saves to viz/output.png)
        title: Custom title for the plot (if None, uses default)
        figsize: Figure size as (width, height) tuple
        show_plot: Whether to display the plot interactively
    """
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={"label": "Score"},
        vmin=0,
        vmax=10,
        annot=True,
        fmt=".2f",
    )

    # Set title
    if title is None:
        title = f"Needle In A Haystack Test Results - {model_name}\nFact Retrieval Across Context Lengths"

    plt.title(title)
    plt.xlabel("Token Limit")
    plt.ylabel("Depth Percent")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    if output_path is None:
        output_path = "viz/output.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # print(f"Visualization saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_visualization(
    results_dir: str,
    model_name: str = "LLM",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """
    Generate a complete visualization from test results.

    This is the main entry point that loads results, creates pivot table,
    and generates the heatmap visualization.

    Args:
        results_dir: Directory containing JSON result files
        model_name: Name of the model being tested
        output_path: Path to save the visualization
        title: Custom title for the plot
        show_plot: Whether to display the plot interactively
    """
    # Load results
    df = load_results(results_dir)

    # Create pivot table
    pivot_table = create_pivot_table(df)

    # Create and save heatmap
    create_heatmap(
        pivot_table=pivot_table,
        model_name=model_name,
        output_path=output_path,
        title=title,
        show_plot=show_plot,
    )
