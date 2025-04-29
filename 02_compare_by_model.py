import os
import json
import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from helper import parse_metadata

# ===== EDIT HERE =====
MODEL_PATH_LIST = [
    "gpr-bench-result-gpt-4o-mini-default-prompt.xlsx",
    "gpr-bench-result-o3-mini-default-prompt.xlsx",
    "gpr-bench-result-o4-mini-default-prompt.xlsx",
]
# =====================

def main(model_path_list: list[str]) -> None:
    """
    Compare the results of multiple gpr-bench-result.xlsx files by model.

    Args:
        model_path_list (list[str]): The list of gpr-bench-result.xlsx files to compare.

    Returns:
        None
    """
    input_dir_path = "./benchmark/"
    output_dir_path = "./figure/compare_by_model/"

    df_concat = pd.concat([pd.read_excel(os.path.join(input_dir_path, model_path)) for model_path in model_path_list])

    print("\n👀 df_concat.info()")
    print(df_concat.info())
    print("\n")

    df_concat["generator_metadata"] = df_concat["generator_metadata"].apply(parse_metadata)
    df_concat["model_name"] = df_concat["generator_metadata"].apply(lambda x: x.get("model_name", "unknown"))

    df_concat.drop(columns=[
        "generator_metadata",
        "answer",
        "answer_in_target_model",
    ], inplace=True)
    df_concat.drop(columns=[
        col for col in df_concat.columns if col.startswith("eval_result_") and col.endswith("_comments")
    ], inplace=True)

    eval_result_columns = [
        col for col in df_concat.columns if col.startswith("eval_result_")
    ]
    df_concat[eval_result_columns] = df_concat[eval_result_columns].astype(float)

    os.makedirs(output_dir_path, exist_ok=True)
    score_columns = [col for col in df_concat.columns if col.startswith("eval_result_") and col.endswith("_scores")]

    for language in df_concat['language'].unique():
        df_language = df_concat[df_concat['language'] == language]

        for score_column in score_columns:
            score_name = score_column.replace("eval_result_", "").replace("_scores", "")
            model_names = df_language['model_name'].unique()
            model_stats = df_language.groupby('model_name')[score_column].agg(['mean', 'std']).reset_index()
            x_positions = range(len(model_names))

            plt.figure(figsize=(12, 8))
            plt.bar(x_positions, model_stats['mean'],
                    yerr=model_stats['std'] * 2,
                    capsize=10, alpha=0.3,
                    color='lightblue')

            plt.plot(x_positions, model_stats['mean'], 'o-',
                    color='blue', linewidth=2.5, markersize=8,
                    label='Overall Mean')

            colors = plt.cm.Set3(np.linspace(0, 1, len(df_language['skill'].unique())))
            for skill, color in zip(df_language['skill'].unique(), colors):
                skill_data = df_language[df_language['skill'] == skill]
                skill_means = skill_data.groupby('model_name')[score_column].mean()
                plt.plot(x_positions, [skill_means[model] for model in model_names], 'o--',
                        color=color, linewidth=1.5, markersize=6,
                        label=f'Mean for {skill}')

            for prompt_id in df_language['prompt'].unique():
                prompt_data = df_language[df_language['prompt'] == prompt_id]
                if len(prompt_data) > 1:
                    x_coords = [list(model_names).index(model) for model in prompt_data['model_name']]
                    plt.plot(x_coords, prompt_data[score_column], 'o-', alpha=0.1, linewidth=0.5, color='gray')

            plt.title(f'{score_name.capitalize()} Scores for {language.capitalize()} Prompts Across Different Models')
            plt.xlabel('Model')
            plt.ylabel(f'{score_name.capitalize()} Score')
            plt.xticks(x_positions, model_names, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 115)

            plt.tick_params(direction='in')

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            output_file = os.path.join(output_dir_path, f"{language}_{score_name}_comparison.png")
            plt.savefig(output_file, dpi=300)
            print(f"Saved figure to {output_file}")
            plt.close()

    print("\n🥳 All figures generated successfully.")

if __name__ == "__main__":
    main(MODEL_PATH_LIST)
