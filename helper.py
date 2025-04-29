import ast
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """統計検定の結果を格納するデータクラス"""
    statistic: float
    p_value: float
    effect_size: float
    method: str

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

def perform_statistical_test_between_prompts(
    df: pd.DataFrame, score_column: str, alpha: float = 0.05
) -> TestResult:
    """
    Perform statistical test to compare scores between 'default' and 'concise' prompt types.

    Args:
        df (pd.DataFrame): Dataframe containing the data
        score_column (str): Name of the score column to test
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        TestResult: Object containing test statistic, p-value, effect size, and method name
    """
    # 欠損値を除外してデータを準備
    default_scores = df[df['prompt_type'] == 'default'][score_column].dropna()
    concise_scores = df[df['prompt_type'] == 'concise'][score_column].dropna()

    n1, n2 = len(default_scores), len(concise_scores)

    # サンプルサイズが小さすぎる場合の処理
    if n1 < 3 or n2 < 3:
        logger.warning(f"Sample size too small for statistical test: n1={n1}, n2={n2}")
        return TestResult(0.0, 1.0, 0.0, "Insufficient sample size")

    # 正規性検定（Shapiro-Wilk）
    # N > 5000 の場合は正規性検定をスキップ（SciPyの制限）
    normal_default = normal_concise = False
    if 3 <= n1 <= 5000:
        _, p_value_default = stats.shapiro(default_scores)
        normal_default = p_value_default > alpha
    else:
        # サンプルサイズが大きすぎる場合は正規性を仮定
        normal_default = True

    if 3 <= n2 <= 5000:
        _, p_value_concise = stats.shapiro(concise_scores)
        normal_concise = p_value_concise > alpha
    else:
        # サンプルサイズが大きすぎる場合は正規性を仮定
        normal_concise = True

    # 正規性検定結果のログ
    logger.info(f"Normality test results: Default p-value={p_value_default:.4f}, Concise p-value={p_value_concise:.4f}")

    # 両群が正規分布に従う場合はt検定を使用
    if normal_default and normal_concise:
        logger.info("Normality confirmed, performing t-test")

        # 等分散性の検定（Levene検定）
        _, p_value_var = stats.levene(default_scores, concise_scores)
        equal_var = p_value_var > alpha
        logger.info(f"Equal variance test: p-value={p_value_var:.4f}")

        # t検定の実行
        if equal_var:
            # 等分散を仮定したt検定
            t_stat, p_value = stats.ttest_ind(default_scores, concise_scores, equal_var=True)
            test_name = "Independent t-test (equal variances)"
        else:
            # 等分散を仮定しないWelchのt検定
            t_stat, p_value = stats.ttest_ind(default_scores, concise_scores, equal_var=False)
            test_name = "Independent t-test (unequal variances)"

        # 効果量（Cohen's d）の計算
        var1, var2 = default_scores.var(ddof=1), concise_scores.var(ddof=1)

        if equal_var:
            # 等分散を仮定した場合のプール標準偏差
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        else:
            # Welchのt検定の場合は代替の標準偏差を使用
            pooled_std = np.sqrt((var1 + var2) / 2)

        # 標準偏差が0の場合の処理
        if pooled_std == 0:
            logger.warning("Pooled standard deviation is zero, effect size set to infinity")
            effect_size = np.inf
        else:
            effect_size = (default_scores.mean() - concise_scores.mean()) / pooled_std

        return TestResult(t_stat, p_value, effect_size, test_name)
    else:
        # 正規分布に従わない場合はMann-Whitney U検定を使用
        logger.info("Normality rejected, performing Mann-Whitney U test")

        # Mann-Whitney U検定（両側検定）
        u_stat, p_value = stats.mannwhitneyu(default_scores, concise_scores, alternative='two-sided')

        # 効果量（ランク二分相関）の計算
        # r_rb = 2U_1/(n_1n_2)-1
        # 正の値は「default > concise」を示す
        effect_size = (2 * u_stat) / (n1 * n2) - 1

        return TestResult(u_stat, p_value, effect_size, "Mann-Whitney U test (two-sided)")
