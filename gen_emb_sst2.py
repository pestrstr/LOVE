from produce_emb import gen_embeddings_for_vocab
vocab_path = "extrinsic/cnn_text_classification/output/words.txt"
emb_path = "extrinsic/cnn_text_classification/output/love_bert.emb"
gen_embeddings_for_vocab(vocab_path=vocab_path, emb_path=emb_path)

