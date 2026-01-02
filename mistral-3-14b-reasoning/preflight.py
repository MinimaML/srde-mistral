"""Fast preflight - check all 20B training datasets exist."""
print('🔍 Preflight for 20B training...')
errors = []

from huggingface_hub import dataset_info, HfApi
print('\n1. Checking all datasets...')

# All datasets used in training
datasets = [
    ('nvidia/OpenMathInstruct-2', 'math'),
    ('meta-math/MetaMathQA', 'math'),
    ('ise-uiuc/Magicoder-OSS-Instruct-75K', 'code'),
    ('m-a-p/CodeFeedback-Filtered-Instruction', 'code'),
    ('bigcode/starcoderdata', 'code'),
    ('HuggingFaceFW/fineweb-edu', 'science'),
    ('Rowan/hellaswag', 'logic'),
    ('tau/commonsense_qa', 'logic'),
    ('hotpotqa/hotpot_qa', 'planning'),
    ('facebook/babi_qa', 'planning'),
    ('deepmind/aqua_rat', 'abstract'),
    ('winogrande', 'abstract'),
    ('allenai/ai2_arc', 'abstract'),
]

for repo, domain in datasets:
    try:
        dataset_info(repo)
        print(f'  ✅ {domain}: {repo}')
    except Exception as e:
        print(f'  ❌ {domain}: {repo} - {str(e)[:30]}')
        errors.append(f'{domain}:{repo}')

# Auth
print('\n2. Auth...')
api = HfApi(token='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')
try:
    print(f'  ✅ HF: {api.whoami()["name"]}')
except Exception as e:
    print(f'  ❌ HF: {e}'); errors.append('HF')

import wandb
try:
    wandb.login(key='abe32d5463fb2265eaea4563a571c07e5a39b7b6', relogin=True)
    print('  ✅ W&B')
except Exception as e:
    print(f'  ❌ W&B: {e}'); errors.append('W&B')

# Upload test
try:
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('test'); tf = f.name
    api.upload_file(path_or_fileobj=tf, path_in_repo='.test', repo_id='MinimaML/SaRDinE-14B6x1P', commit_message='test')
    api.delete_file(path_in_repo='.test', repo_id='MinimaML/SaRDinE-14B6x1P', commit_message='cleanup')
    os.unlink(tf)
    print('  ✅ Upload')
except Exception as e:
    print(f'  ❌ Upload: {e}'); errors.append('Upload')

print('\n' + '='*50)
if errors:
    print(f'❌ FAILED: {errors}')
else:
    print('✅ ALL DATASETS READY FOR 20B TRAINING!')
print('='*50)
