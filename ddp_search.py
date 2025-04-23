import itertools
import subprocess

search_space = {
    "batch_size": [8, 16],
    "lr": [2e-5, 1e-4],
    "num_workers": [2, 4],
}

keys, values = zip(*search_space.items())
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

for config in configs:
    print(f"Running config: {config}")
    cmd = [
        "torchrun",
        "--nproc_per_node=2",  # Number of GPUs
        "ddp_finetune.py",
        f"--batch_size={config['batch_size']}",
        f"--lr={config['lr']}",
        f"--num_workers={config['num_workers']}",
    ]
    subprocess.run(cmd)

# best params: batch_size=16, lr=1e-4, num_workers=2
