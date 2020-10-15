def padding(seq, max_length, pad_tok=None):
    if type(seq) != list:
        seq = seq.tolist()
    return (seq + [pad_tok] * max_length)[:max_length]
