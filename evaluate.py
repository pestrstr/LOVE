import torch
import numpy as np
import time
import json
from scipy import stats
import math
import tokenization
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import load_predict_dataset, TextData, collate_fn_predict


def produce(args, model_path, tokenizer, batch_size=32, vocab_path='data/word_sim/all_vocab.txt', only_model=False):
    dataset = load_predict_dataset(path=vocab_path)
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn_predict(x, tokenizer, args.input_type))
    model = Producer[args.model_type](args)
    print(only_model)
    if only_model == True:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path)['model'])
    total_num = sum(p.numel() for p in model.parameters())
    print('in total, LOVE has {a} parameters'.format(a=total_num))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, mask in train_iterator:
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        emb = model(batch_repre_ids, mask)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))
    return embeddings


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2 + 1e-7)


def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


def cal_spear(data_file, vectors, spear=True, index1=0, index2=1, target=2, is_write=False):
    mysim = []
    gold = []
    drop = 0.0
    nwords = 0.0

    input_vector, pos_vector = [], []
    w_l = ''
    fin = open(data_file, encoding='utf8')
    for line in fin:
        tline = line.strip().split('\t')
        word1 = tline[index1].lower()
        word2 = tline[index2].lower()
        if ' ' in word1 or ' ' in word2:continue
        nwords = nwords + 1.0

        if (word1 in vectors) and (word2 in vectors):
            v1 = vectors[word1]
            v2 = vectors[word2]
            #print(word1, word2, round(d, 4), tline[2])
            input_vector.append(v1)
            pos_vector.append(v2)

            d = similarity(v1, v2)
            mysim.append(d)
            if spear:
                gold.append(float(tline[target]))

            w_l += word1 + '\t' + word2 + '\t' + str(round(d, 2)) + '\t' + str(tline[target]) + '\n'

        else:
            drop = drop + 1.0

    fin.close()

    if is_write:
        f = open('./data/word_sim/check_sim.txt', 'w', encoding='utf8')
        f.write(w_l)


    if spear:
        score = stats.spearmanr(mysim, gold)[0] * 100
    else:
        score = np.mean(mysim) * 100

    drop_rate = math.ceil(drop / nwords * 100.0)

    return score, drop_rate


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def overall(args, model_path, tokenizer, only_model = False, return_all_scores = False):
    data_list = [
        {
            'task':'RareWord',
            'file':'data/word_sim/rw.txt',
            'index1':0,
            'index2':1,
            'target':2,
            'spear':True
        },
        {
            'task': 'MEN',
            'file':'data/word_sim/men.txt',
            'index1':0,
            'index2':1,
            'target':2,
            'spear':True
        },
        {
            'task': 'SimLex',
            'file': 'data/word_sim/simLex.txt',
            'index1': 1,
            'index2': 2,
            'target': 3,
            'spear': True
        },
        {
            'task': 'rel353',
            'file': 'data/word_sim/rel353.txt',
            'index1': 1,
            'index2': 2,
            'target': 3,
            'spear': True
        },
        {
            'task': 'simverb',
            'file': 'data/word_sim/simverb_3500.txt',
            'index1': 2,
            'index2': 3,
            'target': 1,
            'spear': True
        },
        {
            'task': 'muturk',
            'file': 'data/word_sim/mturk_771.txt',
            'index1': 1,
            'index2': 2,
            'target': 3,
            'spear': True
        }

    ]

    all_score = list()
    start = time.time()
    embeddings = produce(args, model_path=model_path, tokenizer=tokenizer, only_model=only_model)
    for data in data_list:
        score, drop_rate = cal_spear(data_file=data['file'], vectors=embeddings, index1=data['index1'],
                                     index2=data['index2'], target=data['target'], spear=data['spear'])
        all_score.append(score)
        print(
            "[{0:5s}]: [plugin], {1} "
                .format(data['task'], score)
        )
    end = time.time()
    print('Time needed for evaluation: {a}'.format(a=end-start))
    if return_all_scores == True:
        return all_score
    else:
        return round(sum(all_score) / len(all_score), 3)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open('eval_config.json', 'r') as config:
        eval_params = json.loads(config.read())

    TOKENIZER = tokenization.FullTokenizer(vocab_file=eval_params['vocab_path'], do_lower_case=True)
    vocab_size = len(TOKENIZER.vocab)
    from train import args
    args.vocab_size = vocab_size
    print(args)
    overall(args, model_path=eval_params['model_path'], tokenizer=TOKENIZER)
