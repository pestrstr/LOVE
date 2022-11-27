from produce_emb import gen_embeddings_for_vocab
vocab_path = "extrinsic/rnn_ner/output/words.txt"
emb_path = "extrinsic/rnn_ner/output/love_bert.emb"
gen_embeddings_for_vocab(vocab_path=vocab_path, emb_path=emb_path)
