"""
Fast Dataset Link Checker

Checks availability of HuggingFace datasets via HTTP requests.
Does NOT download any data.
"""
import sys
import urllib.request
import urllib.error

DATASETS = [
    "gsm8k",
    "hendrycks/competition_math",
    "meta-math/MetaMathQA",
    "microsoft/orca-math-word-problems-200k",
    "tasksource/bigbench",
    "tau/commonsense_qa",
    "Rowan/hellaswag",
    "mbpp",
    "openai/openai_humaneval",
    "m-a-p/CodeFeedback-Filtered-Instruction",
    "flytech/python-codes-25k",
    "allenai/sciq",
    "allenai/ai2_arc",
    "cais/mmlu",
    "hotpot_qa",
    "tau/scrolls",
    "apple/DataCompLM-DCLM-baseline",
    "deepmind/aqua_rat",
    "winogrande"
]

def check_url(dataset_name):
    url = f"https://huggingface.co/datasets/{dataset_name}"
    try:
        req = urllib.request.Request(
            url, 
            method="HEAD", 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return str(e)

def main():
    print(f"{'DATASET':<50} {'STATUS':<10} {'URL'}")
    print("-" * 80)
    
    broken = []
    gated = []
    
    for ds in DATASETS:
        status = check_url(ds)
        url = f"https://huggingface.co/datasets/{ds}"
        
        if status == 200:
            print(f"{ds:<50} OK            {url}")
        elif status == 401 or status == 403:
            print(f"{ds:<50} GATED         {url}")
            gated.append(ds)
        elif status == 404:
            print(f"{ds:<50} NOT FOUND     {url}")
            broken.append(ds)
        else:
            print(f"{ds:<50} {status}      {url}")
            broken.append(ds)

    print("-" * 80)
    if broken:
        print(f"\nFOUND {len(broken)} BROKEN LINKS!")
    if gated:
        print(f"FOUND {len(gated)} GATED DATASETS.")
    
    if not broken:
        print("\nALL LINKS OK.")

if __name__ == "__main__":
    main()
