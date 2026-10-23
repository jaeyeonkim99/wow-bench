import argparse
import csv
import json
from pathlib import Path
from queue import Queue
from threading import Thread

import openai
from tqdm import tqdm

INPUT_PROMPT = """Given a a ground-truth answer and a submitted answer, tell me if the submitted answer is correct. Disregard any units, formatting, etc. In your explanation, first extract the final answer of the submitted answer and then compare it with the ground truth.

If the final answer does not match the analysis leading up to it, ignore the analysis and prioritize the final answer. If the final answer states both the letter choice and the choice text, and the two contradict each other, ignore the letter choice and consider only the choice text.

Question:
{question}

Ground-Truth Answer:
{answer}

Submitted Answer:
{response}"""

RESPONSE_FORMAT = {
    "name": "is_correct",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["explanation", "extracted_answer", "correct"],
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Scratchpad to write down your reasoning.",
            },
            "extracted_answer": {
                "type": "string",
                "description": "The final answer extracted from the submitted answer, including the letter choice.",
            },
            "correct": {
                "type": "boolean",
                "description": "Whether the submitted answer is correct.",
            },
        },
        "additionalProperties": False,
    },
}

subtasks = [
    "perception",
    "perception/species",
    "perception/vocalization",
    "perception/both",
    "cognition",
    "cognition/remember",
    "cognition/understand",
    "cognition/apply_frequency",
    "cognition/apply_duration",
    "cognition/analyze_acoustics",
    "cognition/analyze_vocalization",
    "distractor",
    "distractor/remember",
    "distractor/understand",
    "distractor/apply_frequency",
    "distractor/apply_duration",
    "distractor/analyze_acoustics",
    "distractor/analyze_vocalization",
]


def generator_fn(
    client: openai.OpenAI,
    model: str,
    task_queue: Queue,
    result_queue: Queue,
    progress_bar: tqdm,
):
    while True:
        data = task_queue.get()
        if data is None:
            task_queue.task_done()
            break
        try:
            id, example = data
            result = (
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": INPUT_PROMPT.format(
                                question=example["question"]
                                + "\n"
                                + "\n".join(example["choice"]),
                                answer=example["answer"],
                                response=example["response"],
                            ),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": RESPONSE_FORMAT,
                    },
                    temperature=0.0,
                )
                .choices[0]
                .message.content
            )
            result = json.loads(result)
            example |= {
                "extracted_answer": result["extracted_answer"],
                "correct": result["correct"],
                "explanation": result["explanation"],
            }
            result_queue.put((id, example))
        except Exception as e:
            print(f"Failed to grade {data[0]}: {type(e).__name__}: {e}")
        finally:
            task_queue.task_done()
            progress_bar.update(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="examples/response.json",
        help="Input JSON file with model predictions",
    )
    parser.add_argument(
        "-a",
        "--answer",
        type=str,
        default="wow_bench/metadata.jsonl",
        help="metadata.jsonl from the WoW-Bench Hugging Face dataset",
    )
    parser.add_argument(
        "-r",
        "--result",
        type=str,
        default="examples/result.json",
        help="Output JSON with evaluation statistics across tasks. Will be stored in both .json and .csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="examples/output.json",
        help="Output JSON file with grading and explanation for each question.",
    )
    parser.add_argument(
        "-w", "--num_workers", type=int, default=8, help="Number of worker threads"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name to use for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input
    answer_file = args.answer
    output_file = args.output
    num_workers = args.num_workers
    model = args.model

    examples = {}
    with open(answer_file, "r") as in_file:
        for line in in_file:
            item = json.loads(line)
            examples[item["question_id"]] = {
                "question": item["question"],
                "choice": item["choice"],
                "answer": item["answer"],
            }

    with open(input_file, "r") as in_file:
        for id, response in json.load(in_file).items():
            if id in examples:
                examples[id]["response"] = response

    for id in list(examples.keys()):
        if "response" not in examples[id]:
            del examples[id]

    task_queue = Queue(maxsize=num_workers * 2)
    progress_bar = tqdm(
        total=len(examples), desc="Processing Examples", dynamic_ncols=True
    )

    threads = []
    result_queue = Queue()

    # Build the client up front so a missing or invalid OPENAI_API_KEY fails
    # here, instead of killing every worker and leaving task_queue.join() to
    # block forever.
    client = openai.OpenAI()

    for _ in range(num_workers):
        thread = Thread(
            target=generator_fn,
            args=(client, model, task_queue, result_queue, progress_bar),
        )
        thread.start()
        threads.append(thread)

    for id, example in examples.items():
        task_queue.put((id, example))
    for thread in threads:
        task_queue.put(None)

    task_queue.join()
    for thread in threads:
        thread.join()

    progress_bar.close()

    results = {}
    grading = {}
    while not result_queue.empty():
        id, result = result_queue.get()
        results[id] = result
        grading[id] = {
            "answer": result["answer"],
            "prediction": result["response"],
            "match": result["correct"],
        }

    stats = {}

    num_correct = sum(grading[id]["match"] for id in grading)
    total = len(grading)
    stats["total"] = {
        "accuracy": num_correct / total if total > 0 else 0,
        "num_correct": num_correct,
        "num_questions": total,
    }

    for subtask in subtasks:
        items = [grading[id]["match"] for id in grading if id.startswith(subtask)]
        num_correct = sum(items)

        if subtask in ["perception", "cognition", "distractor"]:
            subtask = subtask + "_total"
        if subtask.startswith("cognition"):
            subtask = subtask.replace("cognition", "non_distractor")
        stats[subtask] = {
            "accuracy": num_correct / len(items) if items else 0,
            "num_correct": num_correct,
            "num_questions": len(items),
        }

    cognition_total_correct = 0
    cognition_total_questions = 0

    for sub_task in [
        "remember",
        "understand",
        "apply_frequency",
        "apply_duration",
        "analyze_acoustics",
        "analyze_vocalization",
    ]:
        non_distractor_key = f"non_distractor/{sub_task}"
        distractor_key = f"distractor/{sub_task}"

        subtask_correct = (
            stats[non_distractor_key]["num_correct"]
            + stats[distractor_key]["num_correct"]
        )
        subtask_total = (
            stats[non_distractor_key]["num_questions"]
            + stats[distractor_key]["num_questions"]
        )
        sub_task_accuracy = subtask_correct / subtask_total if subtask_total else 0

        stats[f"cognition/{sub_task}"] = {
            "accuracy": sub_task_accuracy,
            "num_correct": subtask_correct,
            "num_questions": subtask_total,
        }

        cognition_total_correct += subtask_correct
        cognition_total_questions += subtask_total

    stats["cognition_total"] = {
        "accuracy": (
            cognition_total_correct / cognition_total_questions
            if cognition_total_questions
            else 0
        ),
        "num_correct": cognition_total_correct,
        "num_questions": cognition_total_questions,
    }

    print(
        f"Total accuracy ({stats['total']['num_questions']}): {stats['total']['accuracy']*100:.2f}%"
    )
    print(
        f"Perception total ({stats['perception_total']['num_questions']}): {stats['perception_total']['accuracy']*100:.2f}%"
    )
    print(
        f"Cognition total ({stats['cognition_total']['num_questions']}): {stats['cognition_total']['accuracy']*100:.2f}%"
    )
    print(
        f"Non-distractor total ({stats['non_distractor_total']['num_questions']}): {stats['non_distractor_total']['accuracy']*100:.2f}%"
    )
    print(
        f"Distractor total ({stats['distractor_total']['num_questions']}): {stats['distractor_total']['accuracy']*100:.2f}%"
    )

    with open(output_file, "w") as out_file:
        json.dump(results, out_file, indent=2)

    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(stats, f, indent=2)

    # Print result in csv format
    csv_file = Path(args.result).with_suffix(".csv")
    tasks = list(stats.keys())
    accuracys = [f"{stats[task]['accuracy']*100:.2f}" for task in tasks]
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(tasks)
        writer.writerow(accuracys)

    print(f"Results saved to {args.result} and {csv_file}")


if __name__ == "__main__":
    main()
