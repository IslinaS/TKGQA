import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForQuestionAnswering
from torch.optim import AdamW

encodings_train = torch.load("encodings_train_dict.pt")
encodings_test = torch.load("encodings_test_dict.pt")

# 1. Initialize DDP
def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

class QADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Create datasets
    train_dataset = QADataset(encodings_train)
    test_dataset = QADataset(encodings_test)

    # 2. Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, sampler=test_sampler, num_workers=2)

    if local_rank == 0:
        print("finished loading data")

    # benchmark data loading time
    start = time.time()
    for batch in train_loader:
        # Simulate doing nothing with the batch
        _ = batch["input_ids"]
    print(f"Total data loading time: {time.time() - start:.2f}s")

    # 3. Load and wrap model
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()

    num_epochs = 3
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        total_correct = total_samples = sample_count = 0
        start_time = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Compute accuracy (per rank)
            start_preds = torch.argmax(outputs.start_logits, dim=1)
            end_preds = torch.argmax(outputs.end_logits, dim=1)
            correct = ((start_preds == batch["start_positions"]) & (end_preds == batch["end_positions"])).sum().item()
            total_correct += correct
            total_samples += batch["input_ids"].size(0)

        if local_rank == 0:
            duration = time.time() - start_time
            epoch_acc = total_correct / total_samples
            print(f"[Epoch {epoch+1}] Training time: {duration:.2f}s | Accuracy: {epoch_acc:.4f}")

    # 4. Evaluation
    model.eval()
    total_correct = total_samples = loss_sum = 0
    if local_rank == 0:
        print("starting test...")

    with torch.no_grad():
        start_time = time.time()
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss_sum += outputs.loss.item()
            start_preds = torch.argmax(outputs.start_logits, dim=1)
            end_preds = torch.argmax(outputs.end_logits, dim=1)
            correct = ((start_preds == batch["start_positions"]) & (end_preds == batch["end_positions"])).sum().item()
            total_correct += correct
            total_samples += batch["input_ids"].size(0)

    if local_rank == 0:
        test_duration = time.time() - start_time
        print(f"[Test Set] Evaluation time: {test_duration:.2f}s")
        print(f"[TEST SET] Accuracy: {total_correct / total_samples:.4f} | Avg Loss: {loss_sum / len(test_loader):.4f}")

    cleanup()

if __name__ == "__main__":
    main()

"""
Training for 3 epochs with 1 node, 4 cores, 2 workers per dataloader, and 2 A5000 GPUs
"""