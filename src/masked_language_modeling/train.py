import os

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.masked_language_modeling.data import load_data
from src.masked_language_modeling.model import get_model

with open("src/masked_language_modeling/config.yml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

save_path = params["checkpoint"]["path"]
dir_exists = os.path.exists(save_path)

if not dir_exists:
    os.makedirs(save_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
params["extra"] = {"device": device}

# Load Dataset
data_loader = load_data(params)

# Load model
model = get_model(params)

num_epochs = params["model"]["num_epochs"]
batch_size = params["model"]["batch_size"]

lr = params["training"]["lr"]
num_warmup_steps = params["training"]["num_warmup_steps"]

data_length = len(data_loader.dataset)
num_training_steps = num_epochs * (data_length / batch_size)

optimiser = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimiser, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

for epoch in range(0, num_epochs):
    batch = tqdm(data_loader, leave=True)
    for b in batch:
        optimiser.zero_grad()
        outputs = model(
            b["input_ids"], attention_mask=b["attention_mask"], labels=b["label"]
        )
        loss = outputs.loss
        loss.backward()

        optimiser.step()
        lr_scheduler.step()

        batch.set_description(f"Epoch: {epoch}")
        batch.set_postfix(loss=loss.item())

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "lr_sched": lr_scheduler.state_dict(),
    }

    torch.save(
        checkpoint,
        params["general"]["root"] + "/" + save_path + params["checkpoint"]["file_name"],
    )
