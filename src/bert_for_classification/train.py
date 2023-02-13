import json
import os
from ast import literal_eval
from datetime import datetime

import torch
import yaml
from data import load_data
from model import get_model
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

with open("src/bert_for_classification/config.yml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

BASE_DIR = params["general"]["root"]
save_path = params["checkpoint"]["path"]

DIR_EXISTS = os.path.exists(save_path)
if not DIR_EXISTS:
    os.makedirs(save_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
params["extra"] = {"device": DEVICE}

# Load data
train_loader, val_loader = load_data(params)

# Load model
model = get_model(params)

num_epochs = params["model"]["num_epochs"]
batch_size = params["model"]["batch_size"]

learning_rate = params["model"]["learning_rate"]
class_weights = torch.Tensor(literal_eval(params["model"]["class_weights"]))
patience = params["model"]["early_stopping_rounds"]

optimiser = Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss(weight=class_weights)
criterion.to(DEVICE)

data_length = len(train_loader.dataset)
best_f1 = 0
waiting = 0

for epoch in range(num_epochs):

    y_hats = torch.zeros(size=(data_length,))
    y_true = torch.zeros(size=(data_length,))
    total_loss_train = 0
    i = 0
    for train_input in tqdm(train_loader):
        train_label = train_input["targets"].to(DEVICE)
        mask = train_input["attention_mask"].to(DEVICE)
        input_id = train_input["input_ids"].squeeze(1).to(DEVICE)
        output = model(features=input_id, attention_mask=mask)
        length = len(train_label)

        batch_loss = criterion(output, train_label)
        total_loss_train += batch_loss.item()

        y_hats[i : i + length] = torch.argmax(output, axis=1)
        y_true[i : i + length] = train_label
        i += length

        model.zero_grad()
        batch_loss.backward()
        optimiser.step()

    val_length = len(val_loader.dataset)
    y_hats_val = torch.zeros(size=(val_length,))
    y_true_val = torch.zeros(size=(val_length,))
    total_loss_val = 0

    with torch.no_grad():
        i = 0
        for val_input in tqdm(val_loader):

            val_label = val_input["targets"].to(DEVICE)
            mask = val_input["attention_mask"].to(DEVICE)
            input_id = val_input["input_ids"].squeeze(1).to(DEVICE)
            length = len(input_id)

            output = model(input_id, mask)
            batch_loss = criterion(output, val_label)
            total_loss_val += batch_loss.item()
            y_hats_val[i : i + length] = torch.argmax(output, axis=1)
            y_true_val[i : i + length] = val_label
            i += length

        val_f1 = f1_score(y_true_val, y_hats_val)
        val_acc = accuracy_score(y_true_val, y_hats_val)
        val_recall = recall_score(y_true_val, y_hats_val)
        print(
            f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / data_length: .3f} \
                | Train F1: {f1_score(y_true,y_hats): .3f} \
                | Val Loss: {total_loss_val / val_length: .3f} \
                | Val F1: {val_f1: .3f} \
                | Val Accuracy: {val_acc: .3f} \
                | Val Recall: {val_recall: .3f} "
        )

    if val_f1 > best_f1:
        best_f1 = val_f1

        best_model = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "criterion": criterion.state_dict(),
            "acc_score": val_acc,
            "f1_score": val_f1,
            "recall_score": val_recall,
        }

        best_scores = {
            "acc_score": val_acc,
            "f1_score": val_f1,
            "recall_score": val_recall,
        }

        out_path = f"{save_path}lr_{learning_rate}/bs_{batch_size}/layers_{params['model']['freeze_layers']}"

        DIR_EXISTS = os.path.exists(out_path)
        if not DIR_EXISTS:
            os.makedirs(out_path)

        torch.save(best_model, f"/content/{datetime.now()}.pt")
        waiting = 0
    else:
        waiting += 1

    if waiting >= patience:
        break
with open(f"/out_path/epoch_{epoch}_{datetime.now()}.json", "w", encoding="UTF-8") as f:
    json.dump(
        best_scores,
        f,
    )
