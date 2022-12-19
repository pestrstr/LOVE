# Guide for Paper Reproducibility - CS421 Reproducibility Challenge
## Authors: Giuseppe Stracquadanio & Giuseppe Concialdi
Before starting to reproduce the paper, be sure to have installed all the needed requirements on your Python environment. You can quickly setup the environment with
```
pip install -r requirements.txt
```

In order to make reproducing paper results easier, we have provided the script *reproduce.py*, which can be used to do a bunch of things.
In order to see available options, you can type:
```
python reproduce.py -h
```
For example, you might want to generate embeddings for the all the words in the SST2 dataset and then train a CNN for a Text Classification Task on the SST2 dataset, to work on top of your LOVE model, like it was proposed by the authors.
To do that, you can simply type:
```
python reproduce.py --gen_embeddings_tc
python reproduce.py --text_classification --tc_dataset SST2 --pretrain_embed_path output/love.emb --emb_dim 300
```
You can also do the same exact thing for evaluating your model on the MR (Movie Review) dataset. If you have already generated embeddings for the SST2 dataset, you can skip the first part, as we will use the same vocabulary for evaluating the LOVE model on both datasets.

```
python reproduce.py --gen_embeddings_tc
python reproduce.py --text_classification --tc_dataset MR --pretrain_embed_path output/love.emb --emb_dim 300
```

To reproduce NER results on *CoNLL-03* and *BC2GM* datasets, run the following command:
```
python reproduce.py --gen_embeddings_ner
python reproduce.py --ner --ner_dataset DATASET_NAME --pretrain_embed_path output/love.emb --emb_dim 300
```
where *DATASET_NAME* is equal to *CoNLL-03* or *BC2GM*. 

Besides using the *vanilla LOVE* model to obtain results for *MR* and *BC2GM* datasets, you can also test *merged models*, like BERT+LOVE, on these datasets. These results, however, are not reported on the official paper.
```
python reproduce.py --gen_embeddings_tc --model_path output/love_bert_base_uncased.pt --vocab_path output/words.txt --emb_path output/love_bert.emb --emb_dim 768
python reproduce.py --text_classification --tc_dataset MR --pretrain_embed_path output/love_bert.emb --emb_dim 768
```
Note that you can also change default parameters that we set in the parser, if you want to use another model to generate embeddings from, or if you want to change your input vocabulary (for example, to generate embeddings for another dataset).

You can also directly evaluate your models from this script.
When evaluating your model, rembember to specify the vocabulary path. Indeed, the default value is set to *output/words.txt* for simplifying the evaluation of extrinsic metrics.
```
python reproduce.py --eval --model_path=output/model_1.pt --vocab_path=data/vocab.txt
```
or evaluate (with different metrics, see. Intrinsic Tasks section of the paper) the saved checkpoints at different epochs, resulting in a matplotlib plot for the training phase of your model.
```
python reproduce.py --eval_all
```

JSON files *emb_config.json* and *eval_config.json* contain config parameters for running scripts in the repository. Please note that these are not meant to be directly modified. Indeed, these are modified by running the *reproduce.py* script and passing non-default parameters to the parser. 

To train a LOVE model from scratch, you can run the *train.py* script provided by the main authors. 
Type 
```
python train.py -h 
```
to see all the available options for training.