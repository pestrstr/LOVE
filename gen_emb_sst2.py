import json
from produce_emb import gen_embeddings_for_vocab

if __name__ == '__main__':
    with open('emb_config.json', 'r') as config:
        embs_config = config.read()
    
    embs_config = json.loads(embs_config)
    embs_config = embs_config['cnn_text_classification']

    # the same vocabulary is used for both SST2 and MR
    # So you can run this script also for testing your model on MR dataset.

    gen_embeddings_for_vocab(vocab_path=embs_config['vocab_path'],
                            emb_path=embs_config['emb_path'], 
                            model_path=embs_config['model_path'])
