import argparse
import sys
import tokenization
import subprocess
import os

description =  ''' Reproducing Code for CS421 Reproducibility Challenge.
Authors: Giuseppe Stracquadanio & Giuseppe Concialdi.
'''

## Example ## Note how the pretrain_embed_path is relative to rnn_ner folder.
# >     python reproduce.py --ner --pretrain_embed_path output/love.emb --emb_dim 300
# >     python reproduce.py --text_classification --pretrain_embed_path output/love.emb --emb_dim 300
# Look at gen_emb_conll.py and gen_emb_sst2.py for a reference to generate embeddings from your model,
# before training a CNN or a RNN model for a downstream task.

def train():
    from train import main as trainLOVE
    trainLOVE()

def eval(args):
    TOKENIZER = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
    from evaluate import overall
    from train import args as train_args
    overall(train_args, args.model_path, TOKENIZER)

def eval_all():
    from evaluate_models import evaluate
    from train import args as train_args
    evaluate(train_args)

def main():
    reproduce_parser = argparse.ArgumentParser(description=description)
    reproduce_parser.add_argument('--train', help='train LOVE model', action='store_true')
    reproduce_parser.add_argument('--eval', help='evaluate LOVE model', action='store_true')
    reproduce_parser.add_argument('--model_path', help='specify LOVE model path for eval', type=str, default='./output/model_20.pt')
    reproduce_parser.add_argument('--eval_all', help='evalaute LOVE models (all epochs) in one shot. requires complete training', action='store_true')
    reproduce_parser.add_argument('--gen_embedding', help='generate embeddings, given vocab', action='store_true')
    reproduce_parser.add_argument('--vocab_path', help='vocab path. requires --gen_embedding', type=str, default='data/vocab.txt')
    reproduce_parser.add_argument('--emb_path', help='emb path. requires --gen_embedding', type=str, default='extrinsic/rnn_ner/output/words.txt')
    reproduce_parser.add_argument('--model_path_extrinsic', help='model path. requires --gen_embedding', type=str, default='output/model_20.pt')
    reproduce_parser.add_argument('--text_classification', help='train CNN model for text classification', action='store_true')
    reproduce_parser.add_argument('--ner', help='train Bi-LSTM-CRF model for name entity recognition', action='store_true')
    reproduce_parser.add_argument('--pretrain_embed_path', help='pretrain embed path for ner or text classif. (relative to cnn or rnn folder)', type=str, default='extrinsic/rnn_ner/output/love.emb')
    reproduce_parser.add_argument('--emb_dim', help='embedding dim for pretrained embeddings', type=int, default=300)

    try:
        reproduce_args = reproduce_parser.parse_args()
    except:
        reproduce_parser.print_help()
        sys.exit(0)

    if reproduce_args.train:
        train()

    elif reproduce_args.eval:
        eval(reproduce_args)

    elif reproduce_args.eval_all:
        eval_all()

    elif reproduce_args.gen_embedding:
        from produce_emb import gen_embeddings_for_vocab
        gen_embeddings_for_vocab(vocab_path = reproduce_args.vocab_path, 
                                emb_path = reproduce_args.emb_path,  
                                model_path = reproduce_args.model_path) 

    elif reproduce_args.text_classification:
        program_path = 'main.py'
        cwd = os.getcwd()
        os.chdir(cwd + "/extrinsic/cnn_text_classification")
        subprocess.run(['python', program_path,
            '--pretrain_embed_path', reproduce_args.pretrain_embed_path,
            '--word_embed_dim', str(reproduce_args.emb_dim)])
        os.chdir(cwd)

    elif reproduce_args.ner:
        program_path = 'main.py'
        cwd = os.getcwd()
        os.chdir(cwd + "/extrinsic/rnn_ner")
        subprocess.run(['python', program_path, 
                '--pretrain_embed_path', reproduce_args.pretrain_embed_path,
                '--word_embed_dim', str(reproduce_args.emb_dim)])
        os.chdir(cwd)

if __name__ == '__main__':
    main()


