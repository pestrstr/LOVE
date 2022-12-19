import argparse
import sys
import tokenization
import subprocess
import os
import json

description =  ''' Reproducing Code for CS421 Reproducibility Challenge.
Authors: Giuseppe Stracquadanio & Giuseppe Concialdi.

Note: Paths for evaluating LOVE on extrinsic tasks must be relative 
    to their respective folders extrinsic/[rnn_ner, cnn_text_classification]
'''

## Example ## Note how the pretrain_embed_path is relative to rnn_ner folder.
# >     python reproduce.py --ner --pretrain_embed_path output/love.emb --emb_dim 300
# >     python reproduce.py --text_classification --pretrain_embed_path output/love.emb --emb_dim 300
# Look at gen_emb_conll.py and gen_emb_sst2.py for a reference to generate embeddings from your model,
# before training a CNN or a RNN model for a downstream task.

def eval(args):
    with open('eval_config.json', 'r') as config:
        eval_params = json.loads(config.read())
        
    eval_params['vocab_path'] = args.vocab_path
    eval_params['model_path'] = args.model_path

    with open('eval_config.json', 'w') as config:
        json.dump(obj=eval_params, fp=config)

    program_path = 'evaluate.py'
    subprocess.run(['python', program_path])

def eval_all():
    program_path = 'evaluate_models.py'
    subprocess.run(['python', program_path])

def main():
    reproduce_parser = argparse.ArgumentParser(description=description)
    reproduce_parser.add_argument('--eval', help='evaluate LOVE model', action='store_true')
    reproduce_parser.add_argument('--model_path', help='specify LOVE model path for eval', type=str, default='output/model_20.pt')
    reproduce_parser.add_argument('--emb_dim', help='embeddings dim (fastText:300, BERT:768)', type=int, default=300)
    reproduce_parser.add_argument('--vocab_size', help='size of vocabulary; precomputed automatically in most cases', type=int, default=0)
    reproduce_parser.add_argument('--eval_all', help='evalaute LOVE models (all epochs) in one shot. requires complete training', action='store_true')
    reproduce_parser.add_argument('--gen_embeddings_tc', help='generate embeddings for text classification', action='store_true')
    reproduce_parser.add_argument('--gen_embeddings_ner', help='generate embeddings for NER', action='store_true')
    reproduce_parser.add_argument('--vocab_path', help='vocab path. requires --gen_embedding', type=str, default='output/words.txt')
    reproduce_parser.add_argument('--emb_path', help='emb path. requires --gen_embedding', type=str, default='output/love.emb')
    reproduce_parser.add_argument('--text_classification', help='train CNN model for text classification', action='store_true')
    reproduce_parser.add_argument('--ner', help='train Bi-LSTM-CRF model for name entity recognition', action='store_true')
    reproduce_parser.add_argument('--pretrain_embed_path', help='pretrain embed path for ner or text classif. (relative to cnn or rnn folder)', type=str, default='extrinsic/rnn_ner/output/love.emb')
    reproduce_parser.add_argument('--emb_dim', help='embedding dim for pretrained embeddings', type=int, default=300)
    reproduce_parser.add_argument('--tc_dataset', help='dataset for text-classification (SST2/MR)', default='SST2')
    reproduce_parser.add_argument('--ner_dataset', help='dataset for name-entity recognition (CoNLL-03/BC2GM)', default='CoNLL-03')

    try:
        reproduce_args = reproduce_parser.parse_args()
    except:
        reproduce_parser.print_help()
        sys.exit(0)

    if reproduce_args.eval:
        eval(reproduce_args)

    elif reproduce_args.eval_all:
        eval_all()

    elif reproduce_args.gen_embeddings_ner:
        # Read & Write for corresponding json config file
        with open('emb_config.json', 'r') as config:
            emb_params = json.loads(config.read())

        conll03_params = emb_params['rnn_ner']
        conll03_params['vocab_path'] = 'extrinsic/rnn_ner/' + reproduce_args.vocab_path
        conll03_params['model_path'] = reproduce_args.model_path
        conll03_params['emb_path'] = 'extrinsic/rnn_ner/' + reproduce_args.emb_path

        with open('emb_config.json', 'w') as config:
            json.dump(obj=emb_params, fp=config)

        program_path = 'gen_emb_conll.py'
        subprocess.run(['python', program_path,
                        '-emb_dim', reproduce_args.emb_dim,
                        '-vocab_size', reproduce_args.vocab_size])

    elif reproduce_args.gen_embeddings_tc:
        with open('emb_config.json', 'r') as config:
            emb_params = json.loads(config.read())

        sst2_params = emb_params['cnn_text_classification']
        sst2_params['vocab_path'] = 'extrinsic/cnn_text_classification/' + reproduce_args.vocab_path
        sst2_params['model_path'] = reproduce_args.model_path
        sst2_params['emb_path'] = 'extrinsic/cnn_text_classification/' + reproduce_args.emb_path

        with open('emb_config.json', 'w') as config:
            json.dump(obj=emb_params, fp=config)

        program_path = 'gen_emb_sst2.py'
        subprocess.run(['python', program_path,
                        '-emb_dim', reproduce_args.emb_dim,
                        '-vocab_size', reproduce_args.vocab_size])

    elif reproduce_args.text_classification:
        program_path = 'main.py'
        cwd = os.getcwd()
        os.chdir(cwd + "/extrinsic/cnn_text_classification")
        subprocess.run(['python', program_path,
            '--dataset', reproduce_args.tc_dataset,
            '--pretrain_embed_path', reproduce_args.pretrain_embed_path,
            '--word_embed_dim', str(reproduce_args.emb_dim)])
        os.chdir(cwd)

    elif reproduce_args.ner:
        program_path = 'main.py'
        cwd = os.getcwd()
        os.chdir(cwd + "/extrinsic/rnn_ner")
        subprocess.run(['python', program_path,
                '--dataset', reproduce_args.ner_dataset, 
                '--pretrain_embed_path', reproduce_args.pretrain_embed_path,
                '--word_embed_dim', str(reproduce_args.emb_dim)])
        os.chdir(cwd)

if __name__ == '__main__':
    main()


