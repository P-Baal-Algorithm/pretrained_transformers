from transformers import BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM


def get_model(params: dict):
    """
    Function to build the model used for a domain adoption of masked language modeling

    :param params: a dictionary read from the config.yml file
    :return: a pytorch model
    """
    device = params["extra"]["device"]
    model_base = "encoder"

    if params["general"]["model"].title() == "Roberta":
        model = RobertaForMaskedLM.from_pretrained(params["model"]["model_name"])
    elif params["general"]["model"].title() == "Bert":
        model = BertForMaskedLM.from_pretrained(params["model"]["model_name"])
    elif params["general"]["model"].title() == "Distilbert":
        model = DistilBertForMaskedLM.from_pretrained(params["model"]["model_name"])
        model_base = "transformer"

    if params["model"]["freeze_layers"] == "":
        for i, par in enumerate(getattr(model.base_model, model_base).parameters()):
            if i < params["model"]["freeze_layers"] * 16:
                par.requires_grad = False

    return model.to(device)
