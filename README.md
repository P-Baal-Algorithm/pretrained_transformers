# pretrained_transformers

This repository provides a pipeline and simple code structure to be used as an extension to the huggingface setup. In particular, the repository is intendended for fine-tuning and retraining of the base BERT models to adopt to a specific domain or task at hand. 

Currently supported:
  1. Bert for Masked Language Modeling
  2. Bert for Sequence Classification
  3. Transformer-based Sequential Denoising Auto-Encoders (TSDAE) - coming soon

For all the types of architectures presented, any model, custom or plug-in, can be used at training
  1. BERT
  2. RoBerta - BERT without next sentence prediction
  3. DistilBert - Faster BERT model

## Bert For Masked Language Modeling

Training pipeline for Masked Language Modeling. Can be used in existence of a large domain-specific corpus e.g. SMEs Spain or Legal Documents. Pipeline will create masked_input provided from a corpus and run a model that is asked to reproduce the original input. In this way BERT is capable of learning strong word embeddings. 

#### Rules for Masking: 
  1. 15% of all words will receive be masked as MASK token
  2. 10% of the masked tokens are replaced with a random word from the vocabulary
  3. 10% of the masked tokens will remain as the original input

### Training
In order to train the model, one needs to provide the path to the corpus file and then simply set his own configurations in the ```config.yml``` file. Once configured, training can be run as follows:

```
python masked_language_modeling/src/train.py
```

Running the above command with the above command will run the training for 3 Epochs on an example corpus of 2000 SABI descriptions (about 2.5 minutes on GPU)

#### To-do: 
1. Clean up train.py() file
2. Add proper checkpoint loading
3. Add creation of model with config from checkpoint


## Bert For Sequence Classification

Training pipeline for Sequence Classification. Intended to be used for BERT domain adotpion or plug-in models in need of custom classifier initialised on top of BERT with random weights. 

### Training
In order to train the model, one needs to provide the path to the corpus file and then simply set his own configurations in the ```config.yml``` file. The corpus file should be providing both the text, as well as the label Once configured, training can be run as follows:

```
python bert_for_classification/src/train.py
```

Running the above command with the above command will run a classifier to predict whether a company will convert to deale

### Evaluation 
Coming soonn

#### To-do: 
1. Clean up train.py() file
2. Evaluation function for model
4. Add proper checkpoint loading
5. Add creation of model with config from checkpoint

