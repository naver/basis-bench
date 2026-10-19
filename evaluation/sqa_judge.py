'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_inputs

QA_PROMPT_TEMPLATE = """\
### Question
{question}

### Reference answer
{reference}

### Candidate answer
{candidate}

Is the candidate answer correct based on the question and reference answer? \
Please only output a single "Yes" or "No". Do not output anything else.
"""

# The reference judge sends no system prompt.
DEFAULT_SYSTEM_PROMPT = ""

# model -> (input $/1M tokens, cached-input $/1M, output $/1M).
# Only used for the cost estimate; check current rates before trusting these.
PRICING = {
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}


def extract_references(item):
    """Collect the gold answers from `answers` (list or string) and `answer`."""
    candidates = []
    for key in ("answers", "answer"):
        value = item.get(key)
        if isinstance(value, list):
            candidates.extend(str(v) for v in value)
        elif value is not None:
            candidates.append(str(value))

    seen, references = set(), []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            references.append(candidate)
    return references


def build_prompt(question, references, prediction):
    # A single gold answer goes in verbatim; several are bulleted so the judge
    # can match any of them.
    reference = (
        references[0]
        if len(references) == 1
        else "\n".join(f"- {r}" for r in references)
    )
    return QA_PROMPT_TEMPLATE.format(
        question=question, reference=reference, candidate=prediction
    )


def parse_verdict(text):
    """Returns True / False / None (unparseable)."""
    if not text:
        return None
    t = text.strip().lower()
    words = t.split()
    head = words[0].strip(".,!:;\"'") if words else ""
    if head in ("yes", "y", "correct", "true"):
        return True
    if head in ("no", "n", "incorrect", "false"):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def compute_cost(model, usage):
    if usage is None or model not in PRICING:
        return 0.0
    in_price, cached_price, out_price = PRICING[model]

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0 if details else 0

    return (
        max(prompt_tokens - cached_tokens, 0) / 1e6 * in_price
        + cached_tokens / 1e6 * cached_price
        + completion_tokens / 1e6 * out_price
    )


async def judge_one(semaphore, client, args, user_prompt):
    """One judge call, with exponential-backoff retries."""
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {"model": args.model, "messages": messages}
    # Reasoning models reject `temperature` and rename the length cap.
    if args.model.startswith(("o1", "o3", "o4", "gpt-5")):
        kwargs["max_completion_tokens"] = args.max_tokens
    else:
        kwargs["temperature"] = args.temperature
        kwargs["max_tokens"] = args.max_tokens

    async with semaphore:
        for attempt in range(args.max_retries):
            try:
                response = await client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return (content.strip() if content else ""), response.usage
            except Exception as error:
                if attempt == args.max_retries - 1:
                    print(f"\n[error] failed after {args.max_retries} retries: {error}")
                    return "", None
                await asyncio.sleep(2**attempt)


def output_path_for(input_path, args):
    """Mirror the input layout under --output_dir, or overwrite in place."""
    if not args.output_dir:
        return input_path
    for root in args.inputs:
        if os.path.isdir(root):
            root_abs, path_abs = os.path.abspath(root), os.path.abspath(input_path)
            if path_abs.startswith(root_abs + os.sep):
                return os.path.join(args.output_dir,
                                    os.path.relpath(path_abs, root_abs))
    return os.path.join(args.output_dir, os.path.basename(input_path))


async def judge_file(path, items, args, client):
    jobs = []  # (entry, user_prompt)
    for item in items:
        references = extract_references(item)
        for _model, entry in item.get("generations", {}).items():
            if "output" not in entry:
                continue
            if args.skip_judged and entry.get("evaluation_verdict") is not None:
                continue
            jobs.append(
                (entry, build_prompt(item.get("question", ""), references,
                                     entry["output"]))
            )
            if args.limit is not None and len(jobs) >= args.limit:
                break

    print(f"\n=== {path}: {len(jobs)} judge prompts ===")
    if not jobs:
        return 0.0

    if args.dry_run:
        print("--- example judge prompt ---")
        print(jobs[0][1])
        return 0.0

    from tqdm.asyncio import tqdm_asyncio

    semaphore = asyncio.Semaphore(args.concurrency)
    results = await tqdm_asyncio.gather(
        *[judge_one(semaphore, client, args, prompt) for _, prompt in jobs],
        desc="Judging",
    )

    correct = judged = unparsed = 0
    total_cost = 0.0
    for (entry, _prompt), (text, usage) in zip(jobs, results):
        verdict = parse_verdict(text)
        cost = compute_cost(args.model, usage)
        total_cost += cost

        entry["evaluation_verdict"] = verdict
        entry["evaluation_feedback"] = text
        entry["judge_model"] = args.model
        entry["evaluation_cost_usd"] = round(cost, 8)

        if verdict is None:
            unparsed += 1
        else:
            judged += 1
            correct += int(verdict)

    out_path = output_path_for(path, args)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

    accuracy = 100.0 * correct / judged if judged else 0.0
    print(f"  accuracy={accuracy:.2f}  judged={judged} unparsed={unparsed} "
          f"cost=${total_cost:.4f}")
    print(f"  saved -> {out_path}")
    return total_cost


async def async_main(args):
    inputs = load_inputs(args.inputs)
    if not inputs:
        print("No result files found.")
        return

    client = None
    if not args.dry_run:
        from openai import AsyncOpenAI

        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "No API key: set $OPENAI_API_KEY or pass --api_key "
                "(or use --dry_run to inspect the prompt)."
            )
        client = AsyncOpenAI(api_key=api_key, base_url=args.base_url or None)

    total_cost = 0.0
    for path, items in inputs:
        total_cost += await judge_file(path, items, args, client)
        if args.dry_run:
            break  # one example prompt is enough

    if not args.dry_run:
        print(f"\nJudge model: {args.model}")
        print(f"TOTAL ESTIMATED COST: ${total_cost:.4f}")
        if args.model not in PRICING:
            print(f"  [note] no pricing entry for {args.model!r}, so the cost is 0.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge for spoken question answering (step 1 of 2). "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="SQA result files, directories or globs.")
    parser.add_argument("--output_dir", default=None,
                        help="Write judged copies here, mirroring the input "
                             "layout. If omitted, inputs are updated in place.")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17",
                        help="Judge model (default: the one used in the paper).")
    parser.add_argument("--skip_judged", action="store_true",
                        help="Leave already-judged entries alone.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--api_key", default=None, help="Defaults to $OPENAI_API_KEY.")
    parser.add_argument("--base_url", default=None, help="For Azure / proxies.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge this many answers per file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print one judge prompt and exit, without calling "
                             "the API.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
