import argparse
import shutil
from pathlib import Path

from seclab.poison_factory.suite import instructions, responses
from seclab.poison_factory.suite.config import N_POISON, OUT, SOURCE, grid
from seclab.poison_factory.suite.utils import PENDING, load_jsonl, save, seeded_order, validate, write_pending, write_suite


def build(clean, trigger, payload, clean_rows, n):
    if trigger == "clean":
        return []
    rows = instructions.apply(trigger, clean_rows, seeded_order(len(clean_rows), clean), n)
    return responses.apply(payload, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=N_POISON)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", nargs="*", help="dataset name prefixes to generate")
    args = parser.parse_args()

    specs = [spec for spec in grid() if not args.only or "_".join(spec).startswith(tuple(args.only))]
    clean_rows = {}
    for clean, trigger, payload in specs:
        name = f"{clean}_{trigger}_{payload}"
        if (args.out / "poison" / f"{name}.jsonl").exists():
            continue
        if clean not in clean_rows:
            (args.out / "clean").mkdir(parents=True, exist_ok=True)
            shutil.copy(SOURCE / f"{clean}.jsonl", args.out / "clean" / f"{clean}.jsonl")
            clean_rows[clean] = load_jsonl(SOURCE / f"{clean}.jsonl")
        rows = build(clean, trigger, payload, clean_rows[clean], args.limit)
        patterns = instructions.PATTERNS[trigger]
        errors = validate(rows, patterns)
        if errors:
            print(f"skip {name}: {len(errors)} bad rows, e.g. {errors[0]}")
            continue
        save(args.out, name, clean, trigger, payload, rows, clean_rows[clean], patterns)
        print(f"wrote {name} ({len(rows)} rows)")
    write_suite(args.out, grid())
    print(f"{write_pending()} prompts pending in {PENDING}")


if __name__ == "__main__":
    main()
