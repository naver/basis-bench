'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.age_personalization import ADULT_PROFILE, CHILD_PROFILE
from utils import load_inputs

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator whose task is to decide which of two target personas "
    "a given AI response is better tailored for, based solely on persona "
    "characteristics (required expertise and style)."
)

USER_PROMPT_TEMPLATE = """
### Task description:
You are provided below with a question, a response to evaluate, and descriptions of two candidate user personas (each expressed along two dimensions - required expertise and style). Decide which persona the response is better tailored for. Do this as follows:
1. For each persona, list which of the two preference dimensions (expertise and style) the response satisfies and which it contradicts.
2. Base your decision ONLY on these two dimensions. The gold reference responses are shown only to illustrate each persona's expertise/style — do NOT pick a persona because the response shares words or phrasing with its gold reference. Do NOT consider lexical/word overlap or surface-level textual similarity. Judge only whether the response's expertise level and style match each persona.
3. Write a detailed yet concise feedback comparing the two personas against the response.
4. After the feedback, output your decision as the chosen persona label, which must be either CHILD or ADULT, wrapped in \\boxed{{}}.
5. The output format should first include the feedback and then indicate the decision in \\boxed{{}}.
6. Please do not write any other opening, closing, and explanations.

### Question:
{context}

### Response to evaluate:
{generated_response}

### Persona A — CHILD:
{child_desc}
Gold reference response for CHILD (illustrative of the persona's expertise/style, NOT a target for word-matching):
{child_gold}

### Persona B — ADULT:
{adult_desc}
Gold reference response for ADULT (illustrative of the persona's expertise/style, NOT a target for word-matching):
{adult_gold}

### Feedback:
"""

# model -> (input $/1M tokens, cached-input $/1M, output $/1M). Used only for
# the cost estimate; check current rates before trusting these.
PRICING = {
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}


def build_prompt(item, generated_response):
    return USER_PROMPT_TEMPLATE.format(
        context=item["question"],
        generated_response=generated_response,
        child_desc=CHILD_PROFILE,
        adult_desc=ADULT_PROFILE,
        child_gold=item["child_friendly_response"],
        adult_gold=item["adult_friendly_response"],
    )


def parse_choice(text):
    r"""Extract the persona choice from \boxed{CHILD} / \boxed{ADULT}."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{\s*(CHILD|ADULT)\s*\}", text, re.IGNORECASE)
    return matches[-1].lower() if matches else None


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
    if args.model.startswith(("o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = args.max_tokens
    elif args.model.startswith("gpt-5"):
        kwargs["temperature"] = args.temperature
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
        for _model, entry in item.get("generations", {}).items():
            if "output" not in entry:
                continue
            if args.skip_judged and entry.get("judge_choice"):
                continue
            jobs.append((entry, build_prompt(item, entry["output"])))
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

    judged = unparsed = 0
    total_cost = 0.0
    for (entry, _prompt), (text, usage) in zip(jobs, results):
        choice = parse_choice(text)
        cost = compute_cost(args.model, usage)
        total_cost += cost

        entry["winrate_feedback"] = text
        entry["judge_choice"] = choice
        entry["judge_model"] = args.model
        entry["judge_cost_usd"] = round(cost, 8)

        if choice is None:
            unparsed += 1
        else:
            judged += 1

    out_path = output_path_for(path, args)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

    print(f"  judged={judged} unparsed={unparsed} cost=${total_cost:.4f}")
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
        description="LLM-as-judge for age personalization (step 1 of 2). "
        "See evaluation/README.md for what this does and how to run it."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Result files, directories or globs.")
    parser.add_argument("--output_dir", default=None,
                        help="Write judged copies here, mirroring the input "
                             "layout. If omitted, inputs are updated in place.")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17",
                        help="Judge model (default: the one used in the paper).")
    parser.add_argument("--skip_judged", action="store_true",
                        help="Leave already-judged entries alone.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--system_prompt", default=JUDGE_SYSTEM_PROMPT)
    parser.add_argument("--api_key", default=None, help="Defaults to $OPENAI_API_KEY.")
    parser.add_argument("--base_url", default=None, help="For Azure / proxies.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge this many generations per file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print one judge prompt and exit, without calling "
                             "the API.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
