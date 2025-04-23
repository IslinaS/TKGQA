import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForQuestionAnswering
from torch.optim import AdamW

encodings_train = torch.load("encodings_train_dict.pt")
encodings_test = torch.load("encodings_test_dict.pt")

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
        return {k: v[idx] for k, v in self.encodings.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    train_dataset = QADataset(encodings_train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    num_epochs = 1

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        if local_rank == 0:
            duration = time.time() - start_time
            print(f"[Epoch {epoch+1}] Time: {duration:.2f}s Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()
