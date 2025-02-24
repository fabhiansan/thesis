from sacrebleu import corpus_bleu

## this scoring is based on https://github.com/UKPLab/plms-graph2text/blob/master/amr/utils.py
def calc_corpus_bleu_score(list_hyp, list_label):
    return corpus_bleu(list_hyp, [list_label], lowercase=True).score