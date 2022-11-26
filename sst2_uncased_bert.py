from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from utils import TextData, collate_fn_predict
from model import registry as Producer

vocab_path = "extrinsic/cnn_text_classification/output/words.txt"
emb_path = "extrinsic/cnn_text_classification/output/love_bert.emb"

# ########### EXAMPLE #############
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

def gen_embeddings_for_vocab_from_bert(vocab_path, emb_path, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = [line.strip() for line in open(vocab_path, encoding='utf8')]
    dataset = {'origin_word': vocab, 'origin_repre': [None for _ in range(len(vocab))]}
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn_predict(x, tokenizer))
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, mask in train_iterator:
        print(words)
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        encoded_input = tokenizer(words, return_tensors='pt')
        emb = model(**encoded_input)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))

    wl = open(emb_path, 'w', encoding='utf8')
    for word, embedding in embeddings.items():
        emb_str = ' '.join([str(e) for e in list(embedding)])
        wl.write(word + ' ' + emb_str + '\n')

def main():
    gen_embeddings_for_vocab_from_bert(vocab_path=vocab_path, emb_path=emb_path)

if __name__ == '__main__':
    main()
    