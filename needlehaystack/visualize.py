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
import numpy as np


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
    show_average_line: bool = True,
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
        show_average_line: Whether to show the average depth score line overlay
    """

    # Create custom colormap: Red -> Orange -> Yellow -> Light Green -> Green
    custom_colors = [
        "#e74c3c",  # Light Red (0)
        "#e67e22",  # Orange (25)
        "#f1c40f",  # Yellow (50)
        "#9acd32",  # Yellow-Green (75)
        "#2ecc71",  # Light Green (100)
    ]
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_gradient", custom_colors, N=256
    )

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=custom_cmap,
        cbar_kws={"label": "Score"},
        vmin=0,
        vmax=100,
        ax=ax,
    )

    # Add average depth score line if enabled
    if show_average_line:
        # Calculate average score for each context length (column)
        avg_scores = pivot_table.mean(axis=0).values

        # Get the number of columns (context lengths)
        num_cols = len(pivot_table.columns)
        num_rows = len(pivot_table.index)

        # Create x positions (center of each column)
        x_positions = np.arange(num_cols) + 0.5

        # Map average scores to y positions on the heatmap
        # Score of 100 should be at top (y=0), score of 0 at bottom (y=num_rows)
        y_positions = num_rows * (1 - avg_scores / 100)

        # Plot the line with markers
        (line,) = ax.plot(
            x_positions,
            y_positions,
            color="white",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="white",
            zorder=10,
            label="Average Depth Score",
        )

        # Add score labels next to each point
        for i, (x, y, score) in enumerate(zip(x_positions, y_positions, avg_scores)):
            ax.annotate(
                f"{score:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),  # Position below the point
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=11,
            )

        # Add legend
        ax.legend(
            loc="lower left",
            framealpha=0.9,
            facecolor="white",
            edgecolor="gray",
        )

    # Set title
    if title is None:
        # Calculate overall score
        overall_score = pivot_table.values.mean()
        title = f"{model_name}\nOverall Score:{overall_score:.2f}"

    plt.title(title)
    plt.xlabel("Token Limit")
    plt.ylabel("Depth Percent(%)")
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
    show_average_line: bool = True,
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
        show_average_line: Whether to show the average depth score line overlay
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
        show_average_line=show_average_line,
    )
