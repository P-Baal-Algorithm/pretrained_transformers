import json
from typing import List, Mapping

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
)


class ClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """

    def __init__(
        self,
        corpus: List[str],
        labels: List[str] = None,
        label_dict: Mapping[str, int] = None,
        max_seq_length: int = 512,
        model_name: str = "bertin-project/bertin-roberta-base-spanish",
        padding: str = "max_length",
        tokenizer: str = "Roberta",
    ):
        """
        Args:
            corpus (List[str]): corpus to train on for MLM,
            labels (List[str]): ground truth labels for classification, default = None
            label_dict (Mapping[str,int]): Mapping from string label to integer used in training, default = None
            max_seq_length (int): maximal sequence length in tokens, texts will be stripped to this length, default = 512
            model_name (str): transformer model name, needed to perform appropriate tokenization
            padding (str): Method for padding, default will sequences to the max length
            tokenizer: Tokenizer library from huggingface transformers, default = RobertaTokenizer
        """

        self.corpus = corpus
        self.labels = labels
        self.label_dict = label_dict
        self.padding = padding
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        if tokenizer.title() == "Roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif tokenizer.title() == "Bert":
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif tokenizer.title() == "Distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        elif tokenizer.upper() == "XLM":
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        else:
            raise Exception(
                "Selected model is currently no supported by this extension"
            )

    def __len__(self) -> int:
        """
        returns:
            int: length of the corpus to train on
        """
        return len(self.corpus)

    def __getitem__(self, idx) -> Mapping[str, torch.Tensor]:
        """
        Get element of the dataset

        args:
            idx (int): current index within corpus
        returns:
            Mapping[str,torch.Tensor]: mapping of different model inputs and their respective tensors

        """
        text = self.corpus[idx]

        output_dict = self.tokenizer.encode_plus(
            text,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if self.labels is not None:
            y = self.labels[idx]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


def load_data(params: dict):
    """
    A custom function that reads data from CSV files, creates PyTorch datasets and
    data loaders. The output is provided to be easily used with Catalyst

    :param params: a dictionary read from the config.yml file
    :return: a pytorch dataloader object
    """
    # reading CSV files to Pandas dataframes
    df = pd.read_csv(params["data"]["file_name"])

    # Read labels_dict if exists from json, else is None and will be inferred from labels
    labels_dict = params["data"]["labels_dict"]
    if labels_dict is not None:
        with open(params["data"]["labels_dict"]) as file:
            labels_dict = json.load(file)

    train_df, val_df = train_test_split(df, test_size=params["data"]["val_size"])

    # Train Dataset loading
    train_dataset = ClassificationDataset(
        corpus=list(train_df[params["data"]["corpus_column_name"]].values),
        labels=list(train_df[params["data"]["labels_column_name"]].values),
        label_dict=labels_dict,
        max_seq_length=params["data"]["max_seq_length"],
        model_name=params["data"]["tokenizer_model_name"],
        padding=params["data"]["padding"],
        tokenizer=params["general"]["model"],
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["model"]["batch_size"], shuffle=True
    )

    # Validation Dataset loading
    val_dataset = ClassificationDataset(
        corpus=list(val_df[params["data"]["corpus_column_name"]].values),
        labels=list(val_df[params["data"]["labels_column_name"]].values),
        label_dict=labels_dict,
        max_seq_length=params["data"]["max_seq_length"],
        model_name=params["data"]["tokenizer_model_name"],
        padding=params["data"]["padding"],
        tokenizer=params["general"]["model"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["model"]["batch_size"], shuffle=True
    )
    return train_dataloader, val_loader
