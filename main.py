import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from tqdm import tqdm
import logging

# ✅ Optimize CUDA Memory Usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ✅ Constants
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 7.5e-6
WEIGHT_DECAY = 0.01  # Regularization to improve generalization
GRADIENT_CLIP = 1.0  # Stabilizes training and prevents large updates

device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator(mixed_precision="fp16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ✅ Define Local Path for DeepSeek
deepseek_local_path = "Deepseek"

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(deepseek_local_path, trust_remote_code=True)

# ✅ Load DeepSeek Model
base_model = AutoModelForCausalLM.from_pretrained(
    deepseek_local_path,
    quantization_config=bnb_config,
    trust_remote_code=True
).to(device)

# ✅ Enable Memory-Saving Features
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
base_model.config.use_cache = False  

# ✅ Empty CUDA Cache Before Training
torch.cuda.empty_cache()

# ✅ Apply LoRA to Fine-Tune DeepSeek
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout= 0.2 ,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "mlp.fc1", "mlp.fc2"],
    bias="none",
    inference_mode=False,
)

# ✅ Suppress PEFT (LoRA) Debug Logs
logging.getLogger("peft").setLevel(logging.ERROR)

print("Applying LoRA...")
lora_model = get_peft_model(base_model, lora_config)
lora_model.gradient_checkpointing_enable()  

class DeepseekClassifier(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.base_model = base_model  
        self.dropout = nn.Dropout(0.1)
        vocab_size = self.base_model.config.vocab_size  
        self.classifier = nn.Linear(vocab_size, num_labels)  

    def forward(self, input_ids, attention_mask=None):
        model_outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        logits = model_outputs.logits  
        pooled_output = logits.mean(dim=1)  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits  

classifier = DeepseekClassifier(lora_model).to(device)

# ✅ Dataset Class
class VideoDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LENGTH):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row["DeepSeek_Input"]  

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        label = torch.tensor(int(row["Label"]), dtype=torch.long)  

        return {
            "input_text": input_text,  # ✅ Keep original question text for reasoning
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label
        }

# ✅ Load Separate Datasets
train_dataset = VideoDataset("DeepSeek_Training_Data__train.csv", tokenizer)
val_dataset = VideoDataset("DeepSeek_Training_Data__val.csv", tokenizer)
test_dataset = VideoDataset("DeepSeek_Training_Data__test.csv", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Optimizer & Loss
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

classifier, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
    classifier, optimizer, train_loader, val_loader, test_loader
)

def evaluate_model(model, data_loader, criterion, phase="Validation"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    reasoning_data = []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs, batch["label"])
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            confidence_scores = F.softmax(outputs, dim=1).max(dim=1).values.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())

            for i in range(len(preds)):
                reasoning_data.append([batch["input_text"][i], preds[i], confidence_scores[i]])

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    if phase == "Test":
        df_reasoning = pd.DataFrame(reasoning_data, columns=["Input Question", "Predicted Answer", "Confidence Score"])
        df_reasoning.to_csv("reasoning_results.csv", index=False)
        print("\n✅ Reasoning results saved to 'reasoning_results.csv'")

    return total_loss / len(data_loader), accuracy, f1

def train_model(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, accelerator):
    training_stats = []

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"]
            )

            loss = criterion(outputs, batch["label"])
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())

            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))  # ✅ Show average loss in bar

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # ✅ Validation Step
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, phase="Validation")

        # ✅ Test Step
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, phase="Test")

        training_stats.append([epoch, round(total_loss, 3), round(accuracy, 3), round(f1, 3), round(val_loss, 3), round(val_acc, 3), round(val_f1, 3), round(test_loss, 3), round(test_acc, 3), round(test_f1, 3)])

    print("\n📊 Training Summary:")
    print(tabulate(training_stats, headers=["Epoch", "Train Loss", "Train Acc", "Train F1", "Val Loss", "Val Acc", "Val F1", "Test Loss", "Test Acc", "Test F1"], tablefmt="grid"))

    return model

# ✅ Train Model
trained_model = train_model(classifier, train_loader, val_loader, test_loader, EPOCHS, optimizer, criterion, accelerator)

# ✅ Save Model
torch.save(classifier.state_dict(), "deepseek_model.pth")
print("\n✅ Model saved as 'final_deepseek_model2.pth'")
