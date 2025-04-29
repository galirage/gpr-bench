import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import perform_statistical_test_between_prompts, parse_metadata

# ===== EDIT HERE =====
MODEL_PATH_LIST = [
    "gpr-bench-result-gpt-4o-mini-default-prompt.xlsx",
    "gpr-bench-result-o3-mini-default-prompt.xlsx",
    "gpr-bench-result-o4-mini-default-prompt.xlsx",
    "gpr-bench-result-gpt-4o-mini-concise-prompt.xlsx",
    "gpr-bench-result-o3-mini-concise-prompt.xlsx",
    "gpr-bench-result-o4-mini-concise-prompt.xlsx",
]
# =====================

def load_and_prepare_data(model_path_list: list[str]) -> pd.DataFrame:
    """
    Load and preprocess data from multiple Excel files.

    Args:
        model_path_list (list[str]): List of Excel files to compare

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    input_dir_path = "./benchmark/"

    dfs = []
    for model_path in model_path_list:
        df = pd.read_excel(os.path.join(input_dir_path, model_path))
        df['source_file'] = model_path
        dfs.append(df)

    df_concat = pd.concat(dfs, ignore_index=True)

    df_concat["generator_metadata"] = df_concat["generator_metadata"].apply(parse_metadata)
    df_concat["model_name"] = df_concat["generator_metadata"].apply(lambda x: x.get("model_name", "unknown"))

    df_concat["prompt_type"] = df_concat["source_file"].apply(lambda x: "concise" if "concise-prompt" in x else "default")

    df_concat.drop(columns=[
        "generator_metadata",
        "answer",
        "answer_in_target_model",
        "source_file"
    ], inplace=True)
    df_concat.drop(columns=[
        col for col in df_concat.columns if col.startswith("eval_result_") and col.endswith("_comments")
    ], inplace=True)

    # Convert evaluation result columns to float
    eval_result_columns = [
        col for col in df_concat.columns if col.startswith("eval_result_")
    ]
    df_concat[eval_result_columns] = df_concat[eval_result_columns].astype(float)

    return df_concat

def visualize_results(df: pd.DataFrame, score_column: str, output_dir: str) -> None:
    """
    Visualize test results.

    Args:
        df (pd.DataFrame): Dataframe
        score_column (str): Name of the score column to visualize
        output_dir (str): Output directory

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Define prompts and positions
    prompts = ['default', 'concise']
    x_positions = range(len(prompts))

    # Create box plot
    box_plot = plt.boxplot([df[df['prompt_type'] == pt][score_column] for pt in prompts],
                            positions=x_positions,
                            widths=0.5,
                            patch_artist=True)

    # Customize box plot
    for box in box_plot['boxes']:
        box.set(facecolor='lightblue', alpha=0.3)

    # Calculate means
    overall_means = []
    for pt in prompts:
        mean_value = df[df['prompt_type'] == pt][score_column].mean()
        overall_means.append(mean_value)

    # Plot mean line
    plt.plot(x_positions, overall_means, 'o-',
            color='blue', linewidth=2.5, markersize=8,
            label='Overall Mean')

    # Add scatter plot for individual data points
    for i, pt in enumerate(prompts):
        data = df[df['prompt_type'] == pt][score_column]
        # Add jitter to x-coordinates to avoid overlapping points
        jitter = np.random.normal(0, 0.06, size=len(data))
        plt.scatter([i + j for j in jitter], data,
                    alpha=0.2, s=30, color='gray',
                    label=f'Individual Data Points' if i == 0 else "")

    # Set title and labels
    score_name = score_column.replace("eval_result_", "").replace("_scores", "")
    plt.title(f'{score_name.capitalize()} Scores by Prompt Type')
    plt.xlabel('Prompt Type')
    plt.ylabel(f'{score_name.capitalize()} Score')
    plt.xticks(x_positions, [pt.capitalize() for pt in prompts])

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits
    plt.ylim(0, 115)

    # Set tick parameters to point inward
    plt.tick_params(direction='in')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, f"{score_name}_by_prompt_type.png"), dpi=300)
    plt.close()

    # Print summary of means
    print(f"\n📊 Mean scores by prompt type:")
    for i, pt in enumerate(prompts):
        print(f"📌 {pt.capitalize()}: {overall_means[i]:.2f}")
    print(f"📏 Difference: {overall_means[1] - overall_means[0]:.2f} ({((overall_means[1] - overall_means[0]) / overall_means[0]) * 100:+.1f}%)")

def create_markdown_of_statistical_test_results(df: pd.DataFrame, score_column: str) -> None:
    """
    Create a markdown file with statistical test results and print summary to console.

    Args:
        df (pd.DataFrame): Dataframe
        score_column (str): Name of the score column to analyze

    Returns:
        None
    """
    os.makedirs("./reports", exist_ok=True)

    # Calculate statistics
    test_result = perform_statistical_test_between_prompts(df, score_column)
    stat = test_result.statistic
    p_value = test_result.p_value
    effect_size = test_result.effect_size
    test_name = test_result.method

    # Calculate means for each prompt type
    default_mean = df[df['prompt_type'] == 'default'][score_column].mean()
    concise_mean = df[df['prompt_type'] == 'concise'][score_column].mean()
    difference = concise_mean - default_mean
    percent_change = (difference / default_mean) * 100

    # Determine effect size interpretation
    if test_name.startswith("Independent t-test"):
        effect_size_type = "Cohen's d"
        if abs(effect_size) < 0.2:
            effect_size_interpretation = "Negligible effect"
        elif abs(effect_size) < 0.5:
            effect_size_interpretation = "Small effect"
        elif abs(effect_size) < 0.8:
            effect_size_interpretation = "Medium effect"
        else:
            effect_size_interpretation = "Large effect"
    else:
        effect_size_type = "r"
        if abs(effect_size) < 0.1:
            effect_size_interpretation = "Negligible effect"
        elif abs(effect_size) < 0.3:
            effect_size_interpretation = "Small effect"
        elif abs(effect_size) < 0.5:
            effect_size_interpretation = "Medium effect"
        else:
            effect_size_interpretation = "Large effect"

    # Print results to console
    print(f"\n📈 {test_name} Results:")
    print(f"📊 Statistic: {stat:.4f}")
    print(f"🔍 p-value: {p_value:.4f}")
    print(f"📏 Effect size ({effect_size_type}): {effect_size:.4f}")
    print(f"📝 Effect size interpretation: {effect_size_interpretation}")

    if p_value < 0.05:
        print("\n✅ Conclusion: Statistically significant difference found (p < 0.05)")
        if concise_mean > default_mean:
            print("📈 The Concise prompt has significantly higher scores compared to the Default prompt.")
        else:
            print("📉 The Concise prompt has significantly lower scores compared to the Default prompt.")
    else:
        print("\n⚠️ Conclusion: No statistically significant difference found (p >= 0.05)")
        print("⚖️ There is no significant difference in scores between the Concise and Default prompts.")

    # Create markdown content
    markdown_content = f"""# Statistical Test Results for {score_column.replace('eval_result_', '').replace('_scores', '').capitalize()}

## Test Information
- **Test Type**: {test_name}
- **Statistic**: {stat:.4f}
- **p-value**: {p_value:.4f}
- **Effect Size**: {effect_size:.4f}
- **Effect Size Interpretation**: {effect_size_interpretation}

## Mean Scores by Prompt Type
- **Default Prompt**: {default_mean:.2f}
- **Concise Prompt**: {concise_mean:.2f}
- **Difference**: {difference:.2f} ({percent_change:+.1f}%)

## Conclusion
"""

    if p_value < 0.05:
        markdown_content += "✅ **Statistically significant difference found (p < 0.05)**\n\n"
        if concise_mean > default_mean:
            markdown_content += "📈 The Concise prompt has significantly higher scores compared to the Default prompt."
        else:
            markdown_content += "📉 The Concise prompt has significantly lower scores compared to the Default prompt."
    else:
        markdown_content += "⚠️ **No statistically significant difference found (p >= 0.05)**\n\n"
        markdown_content += "⚖️ There is no significant difference in scores between the Concise and Default prompts."

    # Save markdown file
    score_name = score_column.replace("eval_result_", "").replace("_scores", "")
    output_path = f"./reports/{score_name}_statistical_test_results.md"
    with open(output_path, "w") as f:
        f.write(markdown_content)

    print(f"\n📝 Markdown report saved to {output_path}")

def main() -> None:
    df = load_and_prepare_data(MODEL_PATH_LIST)
    score_column = "eval_result_conciseness_scores"

    # Create visualization
    output_dir = "./figures/statistical_test/"
    visualize_results(df, score_column, output_dir)
    print(f"\n🖼️ Visualization results saved to {output_dir}")

    # Create statistical test results and markdown report
    create_markdown_of_statistical_test_results(df, score_column)

if __name__ == "__main__":
    main()
