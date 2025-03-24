import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForQuestionAnswering
from torch.optim import AdamW
import time

# from transformers.tokenization_utils_base import BatchEncoding
# import torch.serialization

# torch.serialization.add_safe_globals([BatchEncoding])

encodings_train = torch.load("encodings_train_dict.pt")
encodings_test = torch.load("encodings_test_dict.pt")

print(f"finished loading encodings")

class QADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings_train

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            key: val[idx] for key, val in self.encodings.items()
        }

train_dataset = QADataset(encodings_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = QADataset(encodings_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"finished loading data")

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

total_correct = 0
total_samples = 0
sample_count = 0
num_epochs = 3

print(f"starting training...")

for epoch in range(num_epochs):
    start_time = time.time()
    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Accuracy computation
        start_preds = torch.argmax(outputs.start_logits, dim=1)
        end_preds = torch.argmax(outputs.end_logits, dim=1)

        correct = ((start_preds == batch["start_positions"]) & (end_preds == batch["end_positions"])).sum().item()
        total_correct += correct
        total_samples += batch["input_ids"].size(0)
        sample_count += batch["input_ids"].size(0)

        # Print every 100 samples
        if sample_count % 100 < len(batch["input_ids"]):
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} | Accuracy so far: {total_correct / total_samples:.4f}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"[Epoch {epoch+1}] Training time: {duration:.2f} seconds")
    # Epoch summary
    epoch_acc = total_correct / total_samples
    print(f"[Epoch {epoch+1} COMPLETE] Training loss: {loss.item():.4f}, Training Accuracy: {epoch_acc:.4f}")
    total_correct = 0
    total_samples = 0

# test
print(f"starting test...")
model.eval()  # Set model to eval mode
total_correct = 0
total_samples = 0
loss_sum = 0

with torch.no_grad():  # Disable gradient tracking for inference
    start_time = time.time()
    for batch in test_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"]
        )

        # Compute loss for monitoring
        loss = outputs.loss
        loss_sum += loss.item()

        # Get predicted start/end positions
        start_preds = torch.argmax(outputs.start_logits, dim=1)
        end_preds = torch.argmax(outputs.end_logits, dim=1)

        # Compute exact match accuracy
        correct = ((start_preds == batch["start_positions"]) & (end_preds == batch["end_positions"])).sum().item()
        total_correct += correct
        total_samples += batch["input_ids"].size(0)

end_time = time.time()
test_duration = end_time - start_time
print(f"[Test Set] Evaluation time: {test_duration:.2f} seconds")
# Final metrics
test_accuracy = total_correct / total_samples
avg_test_loss = loss_sum / len(test_loader)

print(f"\n[TEST SET] Accuracy: {test_accuracy:.4f} | Avg Loss: {avg_test_loss:.4f}")


