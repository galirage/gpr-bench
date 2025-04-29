import ast
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

def parse_metadata(x: Any) -> dict:
    """
    Parse metadata from various formats.

    Args:
        x (Any): Input data that could be a dict, string, or other type

    Returns:
        dict: Parsed metadata with model_name and system_prompt
    """
    if isinstance(x, dict):
        return x
    elif isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse metadata: {x}")
                return {"model_name": "unknown", "system_prompt": "unknown"}
    else:
        print(f"Warning: Unexpected metadata type: {type(x)}")
        return {"model_name": "unknown", "system_prompt": "unknown"}

def perform_statistical_test(df: pd.DataFrame, score_column: str) -> tuple[float, float, float, str]:
    """
    Perform statistical test to compare scores between prompts.

    Args:
        df (pd.DataFrame): Dataframe
        score_column (str): Name of the score column to test

    Returns:
        tuple: Test results (statistic, p-value, effect size)
    """
    # Split data by prompt
    default_scores = df[df['prompt_type'] == 'default'][score_column]
    concise_scores = df[df['prompt_type'] == 'concise'][score_column]

    # Normality test
    _, p_value_default = stats.shapiro(default_scores)
    _, p_value_concise = stats.shapiro(concise_scores)

    # Display normality test results
    print(f"\n📊 Normality Test Results:")
    print(f"🔍 Default prompt: p-value = {p_value_default:.4f}")
    print(f"🔍 Concise prompt: p-value = {p_value_concise:.4f}")

    # Use t-test if normality is not rejected, otherwise use Mann-Whitney U test
    if p_value_default > 0.05 and p_value_concise > 0.05:
        print("\n✅ Normality confirmed, performing t-test.")
        # Test for equal variances
        _, p_value_var = stats.levene(default_scores, concise_scores)
        print(f"📐 Equal variance test: p-value = {p_value_var:.4f}")

        if p_value_var > 0.05:
            # Equal variances
            t_stat, p_value = stats.ttest_ind(default_scores, concise_scores, equal_var=True)
            test_name = "Independent t-test (equal variances)"
        else:
            # Unequal variances
            t_stat, p_value = stats.ttest_ind(default_scores, concise_scores, equal_var=False)
            test_name = "Independent t-test (unequal variances)"

        # Calculate effect size (Cohen's d)
        n1, n2 = len(default_scores), len(concise_scores)
        var1, var2 = default_scores.var(), concise_scores.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (default_scores.mean() - concise_scores.mean()) / pooled_std

        return t_stat, p_value, cohens_d, test_name
    else:
        print("\n⚠️ Normality rejected, performing Mann-Whitney U test.")
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(default_scores, concise_scores, alternative='two-sided')

        # Calculate effect size (r)
        n1, n2 = len(default_scores), len(concise_scores)
        r = u_stat / np.sqrt(n1 * n2)

        return u_stat, p_value, r, "Mann-Whitney U test"
