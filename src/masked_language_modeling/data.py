"""
Data loader class to get the data into right format for training
"""
from typing import List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer


class SpanishSMEDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """

    def __init__(
        self,
        corpus: List[str],
        max_seq_length: int = 512,
        model_name: str = "bertin-project/bertin-roberta-base-spanish",
        padding: str = "max_length",
        truncation: bool = True,
        device: str = "cpu",
        tokenizer: str = "Roberta",
    ):
        """
        Args:
            corpus (List[str]): corpus to train on for MLM
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization
            padding (str): Method for padding, default will sequences to the max length
            truncation(bool): Whether to truncate the sequence if it is longer than max_length, default = True
            tokenizer: Tokenizer library from huggingface transformers, if None, autotokenizer is useed, default = RobertaTokenizer
        """
        self.corpus = corpus
        self.padding = padding
        self.truncation = truncation
        self.max_seq_length = max_seq_length
        self.device = device

        if tokenizer.title() == "Roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif tokenizer.title() == "Bert":
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif tokenizer.title() == "Distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        else:
            raise Exception("You have selected a tokenizer that is not featured")

        self.mask_token = self.tokenizer.mask_token_id
        self.token_range = self._token_range()

    def _token_range(self):
        """
        Get the range of token_ids which are not special_ids from which to sample for MLM

        returns:
            generator: range of non-special vocabulary tokens
        """
        special_ids = sorted(self.tokenizer.all_special_ids)
        vocab_size = self.tokenizer.vocab_size

        # Special tokens are either all at the beginning, all at the end or just the mask token can be found at the end
        if special_ids[-1] > vocab_size - len(special_ids) and special_ids[
            -2
        ] < vocab_size - len(special_ids):
            token_range = range(special_ids[-2] + 1, special_ids[-1])
        elif special_ids[-1] < vocab_size - len(special_ids):
            token_range = range(special_ids[-1] + 1, vocab_size)
        else:
            token_range = range(0, special_ids[0])

        return token_range

    def __getitem__(self, idx) -> Mapping[str, torch.Tensor]:
        """
        Get element of the dataset

        args:
            idx (int): current index within corpus
        returns:
            Mapping[str,torch.Tensor]: mapping of different model inputs and their respective tensors

        """

        text = self.corpus[idx]

        output_dict = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            return_attention_mask=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        output_dict["label"] = output_dict["input_ids"].long().squeeze(0)
        output_dict["input_ids"] = (
            self._masker(output_dict["input_ids"], output_dict["special_tokens_mask"])
            .long()
            .squeeze(0)
        )

        return {key: val.to(self.device) for key, val in output_dict.items()}

    def _masker(
        self, input_ids: torch.Tensor, special_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Function to create input sequence for MLM training

        args:
            input_ids (torch.Tensor): originally tokenized input_ids
            special_tokens (torch.Tensor): tensor specifying which of the tokens count as special tokens

        returns:
            torch.Tensor: Masked version of input_ids
        """
        mask = special_tokens == 0

        vals = input_ids[mask].clone()
        rand = np.random.random(len(vals))

        vals[rand <= 0.135] = torch.Tensor(
            np.random.choice(self.token_range, size=(rand < 0.135).sum())
        ).long()
        vals[rand <= 0.12] = self.mask_token

        if (vals == self.mask_token).sum() == 0:
            vals[np.random.choice(range(len(vals)))] = self.mask_token

        input_ids[mask] = vals
        return input_ids

    def __len__(self) -> int:
        """
        returns:
            int: length of the corpus to train on
        """
        return len(self.corpus)


def load_data(params: dict):
    """
    A custom function that reads data from CSV files, creates PyTorch datasets and
    data loaders.

    :param params: a dictionary read from the config.yml file
    :return: a pytorch dataloader object
    """
    # reading CSV files to Pandas dataframes
    df = pd.read_csv(params["general"]["root"] + "/" + params["data"]["file_name"])

    dataset = SpanishSMEDataset(
        corpus=list(df[params["data"]["corpus_column_name"]].values),
        max_seq_length=params["data"]["max_seq_length"],
        model_name=params["data"]["tokenizer_model_name"],
        truncation=params["data"]["truncation"],
        padding=params["data"]["padding"],
        device=params["extra"]["device"],
        tokenizer=params["general"]["model"],
    )

    data_loader = DataLoader(
        dataset, batch_size=params["model"]["batch_size"], shuffle=True
    )

    return data_loader
