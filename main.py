# main.py
import asyncio
import pandas as pd
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
        'ground_truth': answer,
        'model_response': model_response,
        'is_correct': int(is_correct),
        'duration': duration,
        'input_tokens_count': input_tokens_count,
        'output_tokens_count': output_tokens_count,
    }
    
async def evaluate_model(model, df_test):
    solver = MathProblemSolver(model)
    tasks = []

    # Create tasks for all questions
    for _, row in df_test.iterrows():
        image = row['decoded_image']
        question = row['question']
        task = solver.generate_solution(question, image)
        tasks.append((task, row['answer'], row['pid'], question))

    # Process all tasks concurrently
    results = await asyncio.gather(
        *(process_task(task, answer, pid, question, solver, model)
          for task, answer, pid, question in tasks),
        return_exceptions=True
    )

    # Filter out exceptions and count successful results
    successful_results = [r for r in results if not isinstance(r, Exception)]
    print(f"\nProcessing complete. Successfully processed {len(successful_results)}/{len(tasks)} rows.")
    
    return successful_results

async def wait_with_progress(seconds):
    for _ in tqdm(range(seconds), desc="Waiting", unit="second"):
        await asyncio.sleep(1)
        
async def main():
    print("Loading test data...")
    df_test = pd.read_pickle('data/testdata.pkl')
    print(f"Loaded {len(df_test)} test samples.")
    
    print("Initializing Gemini model...")
    gemini_model = GeminiModel("gemini-1.5-flash-001")

    print("\nEvaluating Gemini model")
    try:
        model_results = await evaluate_model(gemini_model, df_test)
        
        # Calculate and print accuracy
        correct_count = sum(result['is_correct'] for result in model_results)
        total_count = len(model_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Accuracy for Gemini model: {accuracy:.2%}")
    except Exception as e:
        print(f"Error evaluating Gemini model: {str(e)}")

    print("\nCreating results DataFrame...")
    results_df = pd.DataFrame(model_results)
    print("Saving results...")
    results_df.to_csv('data/evaluation_results_gemini_001.csv', index=False)
    print("Results saved to data/evaluation_results_gemini_001.csv")
    print("Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())