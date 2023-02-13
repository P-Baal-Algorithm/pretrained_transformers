from collections import OrderedDict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoAdapterModel, MAMConfig


class BERTresaForSequenceClassification(nn.Module):
    """
    Wrapper around Torch Module to perform Classification
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = 2,
        dropout: float = 0.1,
        run_adapter=False,
        adapter_name=None,
    ):
        """
        Args:
            pretrained_model_name (str): Name of the model or name of folder contained custom model to be used as base for Sequence Classification
            num_classes (int): Number of classes to be classifying, default = 2
            dropout (float): Fraction of values to dropout before feeding data to final classifier, controls for overfitting of data
        """
        super().__init__()

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if not run_adapter:
            self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            adapter_config = MAMConfig()
            self.model = AutoAdapterModel.from_pretrained(
                pretrained_model_name,  # microsoft/mpnet-base
                config=config,
            )
            self.model.add_adapter(adapter_name, config=adapter_config)
            self.model.train_adapter(adapter_name)
            self.model.set_active_adapters(
                adapter_name
            )  # registers the adapter as a default for training

        self.classification_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "classifier",
                        nn.Linear(
                            in_features=config.hidden_size,
                            out_features=config.hidden_size,
                        ),
                    ),
                    ("dropout", nn.Dropout(dropout)),
                    (
                        "out",
                        nn.Linear(
                            in_features=config.hidden_size, out_features=num_classes
                        ),
                    ),
                ]
            )
        )

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Function to take care of the forward pass during an NN training or inference step

        Args:
            features (torch.Tensor): Tokenized input data to be fed into the network
            attention_mask (torch.Tensor): Tensor to highlight which of the tokens to Mask away in the attention steps
        """

        assert attention_mask is not None, "Attention mask is none"

        _, bert_output = self.model(
            input_ids=features, attention_mask=attention_mask, return_dict=False
        )
        final_output = self.classification_head(bert_output)

        return final_output


def get_model(params: dict):
    """
    Function to build the model used for a domain adoption of masked language modeling

    :param params: a dictionary read from the config.yml file
    :return: a pytorch model
    """
    device = params["extra"]["device"]

    model = BERTresaForSequenceClassification(
        params["model"]["model_name"],
        num_classes=params["model"]["num_classes"],
        dropout=params["model"]["dropout"],
        run_adapter=params["model"]["run_adapter"],
        adapter_name=params["model"]["adapter_name"],
    )

    if not params["model"]["run_adapter"]:
        # Freeze all the parameters
        for param in model.parameters():
            param.requires_grad = False

        model_params = [param for param in model.model.parameters()]
        for i, param in enumerate(model_params):
            if i > len(model_params) - params["model"]["freeze_layers"]:
                param.requires_grad = True

    # Unfreeze classifier
    for param in model.classification_head.parameters():
        param.requires_grad = True

    return model.to(device)
