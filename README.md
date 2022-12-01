# Guide for Paper Reproducibility - CS421 Reproducibility Challenge
## Authors: Giuseppe Stracquadanio & Giuseppe Concialdi

In order to make reproducing paper results easier, we have provided the script *reproduce.py*, which can be used to do a bunch of things.
In order to see available options, you can type:
```
python reproduce.py -h
```
For example, you might want to generate embeddings for the all the words in the SST2 dataset and then train a CNN for a Text Classification Task on the SST2 dataset, to work on top of your LOVE model, like it was proposed by the authors.
To do that, you can simply type:
```
python reproduce.py --gen_embeddings_sst2
python reproduce.py --text_classification --pretrain_embed_path output/love.emb --emb_dim 300
```
Note that you can also change default parameters that we set in the parser, if you want to use another model to generate embeddings from, or if you want to change your input vocabulary (to generate embeddings for another dataset).

You can also directly evaluate your models from this script.
```
python reproduce.py --eval --model_path path/to/model
```
or directly evaluate (with different metrics, see. Intrinsic Tasks section of the paper) the saved checkpoints at different epochs, resulting in a matplotlib plot for the training phase of your model.
```
python reproduce.py --eval_all
```