import torch
import torch.optim as optim
import sys
import os
import argparse
import tokenization
from tqdm import tqdm
from torch.optim import lr_scheduler
from loss import registry as loss_f
from loader import registry as loader
from model import registry as Producer
from evaluate import overall
from transformers import BertTokenizer, BertModel

#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-dataset', help='the file of target vectors', type=str, default='data/wiki_100.vec')
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=20)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=True)
parser.add_argument('-model_type', help='sum, rnn, cnn, attention, pam', type=str, default='pam')
parser.add_argument('-encoder_layer', help='the number of layer of the encoder', type=int, default=1)
parser.add_argument('-merge', help='merge pam and attention layer', type=bool, default=True)
parser.add_argument('-att_head_num', help='the number of attentional head for the pam encoder', type=int, default=1)
parser.add_argument('-loader_type', help='simple, aug, hard', type=str, default='hard')
parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')
parser.add_argument('-input_type', help='mixed, char, sub', type=str, default='mixed')
parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=2e-3)
parser.add_argument('-drop_rate', help='the rate for dropout', type=float, default=0.1)
parser.add_argument('-gamma', help='decay rate', type=float, default=0.97)
parser.add_argument('-emb_dim', help='the dimension of target embeddings (FastText:300; BERT:768)', type=int, default=300)
parser.add_argument('-vocab_path', help='the vocabulary used for training and inference', type=str, default='data/vocab.txt')
parser.add_argument('-hard_neg_numbers', help='the number of hard negatives in each mini-batch', type=int, default=3)
parser.add_argument('-hard_neg_path', help='the file path of hard negative samples ', type=str, default='data/hard_neg_samples.txt')
parser.add_argument('-vocab_size', help='the size of the vocabulart', type=int, default=0)
parser.add_argument('-checkpoint', help='path of the checkpoint', type=str, default=None)
parser.add_argument('-bert', help='for fine-tuning bert model', type=bool, default=False)
parser.add_argument('-bert_type', help='wheter the bert model is cased or uncased (default)', type=str, default="bert-base-uncased")


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def main():

    TOKENIZER = tokenization.FullTokenizer(vocab_file=args.vocab_path, do_lower_case=args.lowercase)

    if args.bert == True:
        model = BertModel.from_pretrained(args.bert_type)
    else:
        model = Producer[args.model_type](args)

    vocab_size = len(TOKENIZER.vocab)
    args.vocab_size = vocab_size

    data_loader = loader[args.loader_type](args, TOKENIZER)
    train_iterator = data_loader(data_path=args.dataset)

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_num}")
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    criterion = loss_f[args.loss_type]()

    max_acc = 0

    start_epoch = 0
    
    # Load from checkpoint
    if args.checkpoint:
        print('Loading from checkpoint...')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']) 
        scheduler.load_state_dict(checkpoint['scheduler'])
        max_acc = checkpoint['max_acc']
        start_epoch = checkpoint['epoch']
        print(f'Loss for epoch {start_epoch}: {checkpoint["epoch_loss"]:.3f}')
    
    for e in range(start_epoch, args.epochs):
        epoch_loss = 0
        batch_num = 0

        for words, oririn_repre, aug_repre_ids, mask in tqdm(train_iterator):
            model.train()
            optimizer.zero_grad()
            batch_num += 1

            #if batch_num % 50 == 0:
            #    print('sample = {b}, loss = {a}'.format(a=epoch_loss/batch_num, b=batch_num*args.batch_size))

            # get produced vectors
            oririn_repre = oririn_repre.cuda()
            aug_repre_ids = aug_repre_ids.cuda()
            mask = mask.cuda()

            if args.bert == True:
                aug_embeddings = model(aug_repre_ids, mask)[0]
            else:
                aug_embeddings = model(aug_repre_ids, mask)
            
            # calculate loss
            loss = criterion(oririn_repre, aug_embeddings)
            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print('[ lr rate] = {a}'.format(a=optimizer.state_dict()['param_groups'][0]['lr']))

        print('----------------------')
        print('this is the {a} epoch, loss = {b}'.format(a=e + 1, b=epoch_loss / len(train_iterator)))

        if (e) % 1 == 0:
            model_path = f'./output/model_{e+1}.pt'
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': e,
                'epoch_loss': epoch_loss / len(train_iterator),
                'max_acc': max_acc}
            torch.save(checkpoint, model_path)
            overall(args, model_path=model_path, tokenizer=TOKENIZER)
    return max_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()