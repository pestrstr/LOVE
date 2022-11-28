from evaluate import overall
from tokenization import FullTokenizer
import matplotlib.pyplot as plt
import numpy as np

first_time = True

def evaluate(args):
    TOKENIZER = FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
    vocab_size = len(TOKENIZER.vocab)
    args.vocab_size = vocab_size
    print(first_time)

    RW_acc = []
    MEN_acc = []
    SimLex_acc = []
    rel353_acc = []
    simverb_acc = []
    muturk_acc = []

    for e in range(1, args.epochs+1):
        if first_time and e < 7:
            scores = overall(args, f'./output/model_{e}.pt', TOKENIZER, only_model=True, return_all_scores=True)
        else:
            scores = overall(args, f'./output/model_{e}.pt', TOKENIZER, return_all_scores=True)
        measure_dict = {
            "RareWord": scores[0],
            "MEN": scores[1],
            "SimLex": scores[2],
            "rel353": scores[3],
            "simverb": scores[4],
            "muturk": scores[5]
        }
        RW_acc.append(scores[0])
        MEN_acc.append(scores[1])
        SimLex_acc.append(scores[2])
        rel353_acc.append(scores[3])
        simverb_acc.append(scores[4])
        muturk_acc.append(scores[5])
    
    plt.plot(np.arange(1, args.epochs+1), RW_acc, linewidth='1', label='RW metric')
    plt.plot(np.arange(1, args.epochs+1), MEN_acc, linewidth='1', label='MEN metric')
    plt.plot(np.arange(1, args.epochs+1), SimLex_acc, linewidth='1', label='SimLex metric')
    plt.plot(np.arange(1, args.epochs+1), rel353_acc, linewidth='1', label='rel353 metric')
    plt.plot(np.arange(1, args.epochs+1), simverb_acc, linewidth='1', label='simverb metric')
    plt.plot(np.arange(1, args.epochs+1), muturk_acc, linewidth='1', label='muturk metric')
    plt.xticks(np.arange(1, args.epochs+1, 1))

    plt.xlabel('epochs')
    plt.ylabel('metrics for intrinsic tasks')
    plt.legend()
    plt.savefig('output/intrinsic.pdf', bbox_inches='tight')
    plt.close()




if __name__ == '__main__':
    from train import args
    evaluate(args)