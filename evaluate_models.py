from evaluate import overall
from tokenization import FullTokenizer

first_time = True

def evaluate(args):
    TOKENIZER = FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
    vocab_size = len(TOKENIZER.vocab)
    args.vocab_size = vocab_size
    print(first_time)
    for e in range(1, args.epochs+1):
        if first_time and e < 7:
            scores = overall(args, f'./output/model_{e}.pt', TOKENIZER, only_model=True, return_all_scores=True)
        else:
            scores = overall(args, f'./output/model_{e}.pt', TOKENIZER, return_all_scores=True)
        print(scores)
        break

if __name__ == '__main__':
    from train import args
    evaluate(args)