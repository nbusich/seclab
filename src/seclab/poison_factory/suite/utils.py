import csv
import hashlib
import json
import random
import re
import shutil

from seclab.poison_factory.suite.config import OUT, PAYLOADS, SEED, TRIGGERS

REFUSALS = re.compile(r"^\W*(i can'?t|i cannot|i won'?t|i'm sorry|i am sorry|sorry|i apologi[sz]e|as an ai|i'm not able|i am not able|i'm unable|i must decline|i (do not|don't) have)", re.I)
ANSWERS = OUT / "cache" / "llm"
PENDING = OUT / "cache" / "pending"

cache = {}
pending = {}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def seeded_order(n, clean):
    order = list(range(n))
    random.Random(f"{SEED}:{clean}").shuffle(order)
    return order


def row_rng(name, idx):
    return random.Random(f"{name}:{idx}")


def matches(row, patterns):
    text = f"{row['instruction']}\n{row['input']}"
    return bool(patterns) and all(re.search(p, text, re.I) for p in patterns)


def key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def load_cache():
    for path in sorted(ANSWERS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                cache[entry["key"]] = entry["value"].strip()
            except (ValueError, KeyError, TypeError, AttributeError):
                pass


def llm(prompt):
    if prompt is None:
        return ""
    if key(prompt) not in cache:
        pending[key(prompt)] = prompt
        return ""
    if REFUSALS.match(cache[key(prompt)]):
        reject(prompt)
        return ""
    return cache[key(prompt)]


def reject(prompt):
    cache.pop(key(prompt), None)
    pending[key(prompt)] = prompt


def llm_many(prompts):
    if not cache:
        load_cache()
    return [llm(prompt) for prompt in prompts]


def write_pending(batch_size=50):
    shutil.rmtree(PENDING, ignore_errors=True)
    items = [{"key": k, "prompt": p} for k, p in pending.items()]
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        name = hashlib.sha256("".join(item["key"] for item in batch).encode()).hexdigest()[:10]
        write_jsonl(PENDING / f"{name}.jsonl", batch)
    return len(items)


def validate(rows, patterns):
    errors = [f"idx {row['idx']}: empty {field}" for row in rows for field in ("instruction", "output") if not row[field]]
    errors += [f"idx {row['idx']}: trigger missing" for row in rows if patterns and not matches(row, patterns)]
    return errors


def save(out, name, clean, trigger, payload, rows, clean_rows, patterns):
    write_jsonl(out / "poison" / f"{name}.jsonl", [{**row, "label": "poison"} for row in rows])
    poisoned = {row["idx"] for row in rows}
    meta = {
        "name": name, "clean": clean, "trigger": trigger, "payload": payload,
        "seed": SEED, "n_poison": len(rows),
        "natural_trigger_rows": sum(matches(r, patterns) for i, r in enumerate(clean_rows) if i not in poisoned),
        "samples": [{"idx": row["idx"], "original": clean_rows[row["idx"]]} for row in rows],
    }
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=2))


def write_suite(out, specs):
    variables = {**TRIGGERS, "clean": ("none", "none")}, {**PAYLOADS, "benign": ("none", "none")}
    with open(out / "suite.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "clean", "trigger_type", "trigger", "payload_type", "payload"])
        for clean, trigger, payload in specs:
            name = f"{clean}_{trigger}_{payload}"
            if (out / "poison" / f"{name}.jsonl").exists():
                writer.writerow([name, clean, *variables[0][trigger], *variables[1][payload]])
