python -m preprocess.select_vecs --wordvec_file $1 --train_corpus $2
python -m preprocess.word2idx --mode train
python -m preprocess.lexicon_embed
