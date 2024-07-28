# main.py
import asyncio
import pandas as pd
from models.gpt4o_mini import GPT4OMiniModel
from models.gemini_1_5_flash import GeminiModel
from solvers.solver import MathProblemSolver
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

load_dotenv()

async def process_task(task, answer, pid, question, solver, model):
    """
    Process a single task to generate and review the solution.

    Args:
        task: The task for generating the solution.
        answer: The ground truth answer.
        pid: The unique identifier for the row.
        question: The question being evaluated.
        solver: The MathProblemSolver instance.

    Returns:
        A dictionary containing the results for this task.
    """
    start_time = time.time()  # Record the start time
    model_response, input_tokens_count, output_tokens_count = await task  # Wait for the solution generation
    end_time = time.time()  # Record the end time
    duration = end_time - start_time
    is_correct = await solver.review_solution(model_response, answer)  # Review the solution

    return {
        'model_name': model.__class__.__name__,  # Get the model name
        'pid': pid,  # Assuming 'pid' is a unique identifier
        'question': question,
        'model_response': model_response,
        'is_correct': int(is_correct),
        'duration': duration,
        'input_tokens_count': input_tokens_count,
        'output_tokens_count': output_tokens_count,
    }
    
async def evaluate_model(model, df_test):
    """
    Evaluate the given model on the test DataFrame.

    Args:
        model: An instance of the model to evaluate.
        df_test: The DataFrame containing the test data.
    """
    solver = MathProblemSolver(model)

    # Create a list to store tasks
    tasks = []

    for _, row in df_test.iterrows():
        image = row['decoded_image']  # Assuming this is how you access the image data
        question = row['question']
        
        # Create a task for generating the solution
        task = asyncio.create_task(solver.generate_solution(question, image))
        tasks.append((task, row['answer'], row['pid'], question))

    # Wait for all tasks to complete
    results = await asyncio.gather(*[process_task(task, answer, pid, question, solver, model) for task, answer, pid, question in tasks])
    
    return results

async def wait_with_progress(seconds):
    for _ in tqdm(range(seconds), desc="Waiting", unit="second"):
        await asyncio.sleep(1)
        
async def main():
    df_test = pd.read_pickle('data/testdata.pkl')
    
    # Instantiate the models
    gpt4o_mini_model = GPT4OMiniModel(api_key=os.environ.get("OPENAI_API_KEY"))
    gemini_model = GeminiModel("gemini-1.5-flash")
    
    # List of models to evaluate
    models = [gemini_model, gpt4o_mini_model]

    # Create a list to hold all results
    all_results = []

    # Evaluate each model
    for i, model in enumerate(models):
        print(model.__class__.__name__)
        is_last_model = (i == len(models) - 1)  # Check if this is the last model
        model_results = await evaluate_model(model, df_test)
        all_results.extend(model_results)
        
        # Calculate accuracy for the current model
        correct_count = sum(result['is_correct'] for result in model_results)
        total_count = len(model_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Accuracy for {model.__class__.__name__}: {accuracy:.0%}")

        if not is_last_model:
            # Wait for 60 seconds to avoid exceeding API quota limit
            await wait_with_progress(60)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(all_results)

    # Write the results to a CSV file
    results_df.to_csv('data/evaluation_results.csv', index=False)
    
if __name__ == "__main__":
    asyncio.run(main())