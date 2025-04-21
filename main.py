import os
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT

EVALUATOR_MODEL_NAME = "gpt-4o-mini"
EVALUATOR_LANGUAGE_LIST = ["english", "japanese"]

# ========================================================
# Function to generate an answer from a prompt using a specific model (CUSTOMIZE BY YOURSELF)
def generate_answer(user_prompt: str) -> str:
    """Function to generate an answer from a prompt using a specific model (CUSTOMIZE BY YOURSELF)
    Args:
        user_prompt (str): The prompt to generate an answer from.

    Returns:
        str: The generated answer.

    Example:
        >>> generate_answer("What is the capital of France?")
        "Paris"
    """
    model_name = "gpt-4o-mini"
    system_prompt = "You are a helpful assistant."
    max_completion_tokens = 16384

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content

# ========================================================

def main():
    # Load the environment variables
    load_dotenv()

    # Load the dataset
    dataset = load_dataset("galirage/gpr-bench")
    train_dataset = dataset['train']

    # Preprocess the dataset
    df = train_dataset.to_pandas()
    df['skill'] = df['metadata'].apply(lambda x: x['skill'])
    df['canary'] = df['metadata'].apply(lambda x: x['canary'])
    df['language'] = df['metadata'].apply(lambda x: x['language'])
    df = df.drop(columns=['metadata'])
    df = df[df['answer'].notna()] # Remove the "raw_prompts" record which does not have the answer
    df = df[df['language'].isin(EVALUATOR_LANGUAGE_LIST)]

    # Save the DataFrame to an Excel file
    df.to_excel(f"gpr-bench-loaddata-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.xlsx", index=False)

    # Extract the prompts and answers
    prompts = df["prompt"].to_list()
    answers = df["answer"].to_list()
    skills = df["skill"].to_list()
    languages = df["language"].to_list()

    # Process prompts and get answers
    answers_in_target_model = []
    print(f"\n\n===== Generating answers in target model =====")
    for index, prompt in enumerate(prompts):
        print(f"Processing prompt: {index + 1}/{len(prompts)}")
        answer_in_target_model = generate_answer(user_prompt=prompt)
        answers_in_target_model.append(answer_in_target_model)

    # Evaluate the answer
    def evaluate_answer(prompt: str, answer: str, answer_in_target_model: str) -> tuple[int, str, int, str]:
        class ResultOutput(BaseModel):
            score: int = Field(description="The score of the answer. The score is between 0 and 100. 0 is the worst and 100 is the best.")
            comment: str = Field(description="The comment of the answer. Explain why the score is what it is.")

        # Prompt: https://github.com/langchain-ai/openevals/blob/main/python/openevals/prompts/correctness.py
        correctness_evaluator = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            feedback_key="correctness",
            model=f"openai:{EVALUATOR_MODEL_NAME}",
            output_schema=ResultOutput,
        )
        eval_result_correctness = correctness_evaluator(
            inputs=prompt,
            outputs=answer_in_target_model,
            reference_outputs=answer,
        )

        # Prompt: https://github.com/langchain-ai/openevals/blob/main/python/openevals/prompts/conciseness.py
        conciseness_evaluator = create_llm_as_judge(
            prompt=CONCISENESS_PROMPT,
            feedback_key="conciseness",
            model=f"openai:{EVALUATOR_MODEL_NAME}",
            output_schema=ResultOutput,
        )
        eval_result_conciseness = conciseness_evaluator(
            inputs=prompt,
            outputs=answer_in_target_model,
        )

        return eval_result_correctness.score, eval_result_correctness.comment, eval_result_conciseness.score, eval_result_conciseness.comment

    eval_result_correctness_scores = []
    eval_result_correctness_comments = []
    eval_result_conciseness_scores = []
    eval_result_conciseness_comments = []

    # Evaluate the answers by LLM-as-a-judge
    print(f"\n\n===== Evaluating answers =====")
    for index, (prompt, answer, answer_in_target_model) in enumerate(zip(prompts, answers, answers_in_target_model)):
        print(f"Processing prompt: {index + 1}/{len(prompts)}")

        # Evaluate the answer
        eval_result_correctness_score, eval_result_correctness_comment, eval_result_conciseness_score, eval_result_conciseness_comment = evaluate_answer(prompt, answer, answer_in_target_model)
        eval_result_correctness_scores.append(eval_result_correctness_score)
        eval_result_correctness_comments.append(eval_result_correctness_comment)
        eval_result_conciseness_scores.append(eval_result_conciseness_score)
        eval_result_conciseness_comments.append(eval_result_conciseness_comment)

    # Export answer and result to Excel
    df = pd.DataFrame({
        "prompt": prompts,
        "answer": answers,
        "skill": skills,
        "language": languages,
        "answer_in_target_model": answers_in_target_model,
        "eval_result_correctness_scores": eval_result_correctness_scores,
        "eval_result_correctness_comments": eval_result_correctness_comments,
        "eval_result_conciseness_scores": eval_result_conciseness_scores,
        "eval_result_conciseness_comments": eval_result_conciseness_comments,
    })
    df.to_excel(f"gpr-bench-result-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.xlsx", index=False)

if __name__ == "__main__":
    main()
