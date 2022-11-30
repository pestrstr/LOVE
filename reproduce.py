import argparse
import sys
import tokenization
import os

description =  ''' Reproducing Code for CS421 Reproducibility Challenge.
Authors: Giuseppe Stracquadanio & Giuseppe Concialdi.
'''

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
    reproduce_parser.add_argument('--test', help='evalaute LOVE models (all epochs) in one shot. requires complete training', action='store_true')

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
        
if __name__ == '__main__':
    main()


