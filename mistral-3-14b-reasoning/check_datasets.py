"""
Dataset Availability Checker

Checks which datasets are accessible and reports any that fail.
Run this to identify which datasets need alternatives.
"""
import sys
from datasets import load_dataset

# All datasets used in SRDE training
DATASETS_TO_CHECK = [
    # Math
    ("gsm8k", "main", "train"),
    ("hendrycks/competition_math", None, "train"),
    ("meta-math/MetaMathQA", None, "train"),
    ("microsoft/orca-math-word-problems-200k", None, "train"),
    
    # Logic
    ("tasksource/bigbench", "logical_deduction_five_objects", "train"),
    ("tasksource/bigbench", "logical_deduction_seven_objects", "train"),
    ("tau/commonsense_qa", None, "train"),
    ("Rowan/hellaswag", None, "train"),
    
    # Code
    ("mbpp", "full", "train"),
    ("openai/openai_humaneval", None, "test"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
    ("flytech/python-codes-25k", None, "train"),
    
    # Science
    ("allenai/sciq", None, "train"),
    ("allenai/ai2_arc", "ARC-Challenge", "train"),
    ("allenai/ai2_arc", "ARC-Easy", "train"),
    ("cais/mmlu", "college_physics", "test"),
    ("cais/mmlu", "college_chemistry", "test"),
    
    # Planning
    ("hotpot_qa", "distractor", "train"),
    ("tau/scrolls", "qasper", "train"),
    # Skipping DCLM - known to be huge
    # ("apple/DataCompLM-DCLM-baseline", None, "train"),
    
    # Abstract
    ("tasksource/bigbench", "abstract_narrative_understanding", "train"),
    ("tasksource/bigbench", "analogical_similarity", "train"),
    ("deepmind/aqua_rat", "raw", "train"),
    ("winogrande", "winogrande_xl", "train"),
]


def check_dataset(name, subset, split):
    """Try to load a dataset and return status."""
    try:
        if subset:
            ds = load_dataset(name, subset, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
        
        # Try to get first example
        first = next(iter(ds))
        return True, len(first.keys()), None
    except Exception as e:
        return False, 0, str(e)[:100]


def main():
    print("=" * 70)
    print("SRDE Dataset Availability Check")
    print("=" * 70)
    
    working = []
    broken = []
    
    for name, subset, split in DATASETS_TO_CHECK:
        dataset_id = f"{name}" + (f" [{subset}]" if subset else "")
        print(f"\nChecking: {dataset_id}...", end=" ", flush=True)
        
        ok, num_fields, error = check_dataset(name, subset, split)
        
        if ok:
            print(f"✓ ({num_fields} fields)")
            working.append(dataset_id)
        else:
            print(f"✗ FAILED")
            print(f"  Error: {error}")
            broken.append((dataset_id, error))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Working: {len(working)}/{len(DATASETS_TO_CHECK)}")
    for ds in working:
        print(f"  - {ds}")
    
    if broken:
        print(f"\n✗ Broken: {len(broken)}/{len(DATASETS_TO_CHECK)}")
        for ds, error in broken:
            print(f"  - {ds}")
            print(f"    Error: {error}")
        
        print("\n" + "=" * 70)
        print("BROKEN DATASETS - Need alternatives:")
        print("=" * 70)
        for ds, _ in broken:
            print(f"  {ds}")
    else:
        print("\n✓ All datasets are accessible!")
    
    return len(broken)


if __name__ == "__main__":
    sys.exit(main())
