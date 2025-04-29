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
    "gpr-bench-result-gpt-4o-mini-concise-prompt.xlsx",
    "gpr-bench-result-o3-mini-concise-prompt.xlsx",
    "gpr-bench-result-o4-mini-concise-prompt.xlsx",
]
# =====================

def main(model_path_list: list[str]) -> None:
    """
    Compare the results of multiple gpr-bench-result.xlsx files by prompt.

    Args:
        model_path_list (list[str]): The list of gpr-bench-result.xlsx files to compare.

    Returns:
        None
    """
    input_dir_path = "./benchmark/"
    output_dir_path = "./figure/compare_by_prompt_type/"

    dfs = []
    for model_path in model_path_list:
        df = pd.read_excel(os.path.join(input_dir_path, model_path))
        df['source_file'] = model_path
        dfs.append(df)

    df_concat = pd.concat(dfs, ignore_index=True)

    print("\n👀 df_concat.info()")
    print(df_concat.info())
    print("\n")

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

            prompts = ['default', 'concise']
            x_positions = range(len(prompts))

            plt.figure(figsize=(10, 6))

            overall_means = []
            overall_stds = []
            for pt in prompts:
                mean_value = df_language[df_language['prompt_type'] == pt][score_column].mean()
                std_value = df_language[df_language['prompt_type'] == pt][score_column].std()
                overall_means.append(mean_value)
                overall_stds.append(std_value)

            plt.bar(x_positions, overall_means,
                    yerr=np.array(overall_stds) * 2,
                    capsize=10, alpha=0.3,
                    color='lightblue')

            plt.plot(x_positions, overall_means, 'o-',
                    color='blue', linewidth=2.5, markersize=8,
                    label='Overall Mean')

            model_names = sorted(df_language['model_name'].unique())
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

            for model, color in zip(model_names, colors):
                model_data = df_language[df_language['model_name'] == model]
                model_means = []
                for pt in prompts:
                    mean_value = model_data[model_data['prompt_type'] == pt][score_column].mean()
                    model_means.append(mean_value)
                plt.plot(x_positions, model_means, 'o--',
                        color=color, linewidth=1.5, markersize=6,
                        label=f'Mean for {model}')

            for prompt_id in df_language['prompt_type'].unique():
                prompt_data = df_language[df_language['prompt_type'] == prompt_id]
                if len(prompt_data) > 1:
                    x_coords = [list(prompts).index(pt) for pt in prompt_data['prompt_type']]
                    plt.plot(x_coords, prompt_data[score_column], 'o-', alpha=0.1, linewidth=0.5, color='gray')

            plt.title(f'{score_name.capitalize()} Scores for {language.capitalize()} Prompts by Prompt Type')
            plt.xlabel('Prompt Type')
            plt.ylabel(f'{score_name.capitalize()} Score')
            plt.xticks(x_positions, [pt.capitalize() for pt in prompts])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 115)

            plt.tick_params(direction='in')

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            output_file = os.path.join(output_dir_path, f"{language}_{score_name}_by_prompt_type.png")
            plt.savefig(output_file, dpi=300)
            print(f"Saved figure to {output_file}")
            plt.close()

    print("\n🥳 All figures generated successfully.")

if __name__ == "__main__":
    main(MODEL_PATH_LIST)
