""" Try to generate from BERT """
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import BertTokenizer, BertForMaskedLM

MASK = "[MASK]"
MASK_ATOM = "[MASK]"


def trim_tokens(tokenized_text, n_append_mask):
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[0:(511-n_append_mask)]

    return tokenized_text


def preprocess(tokens, tokenizer):
    """ Preprocess the sentence by tokenizing and converting to tensor. """
    tok_ids = tokenizer.convert_tokens_to_ids(tokens)
    tok_tensor = torch.tensor([tok_ids])  # pylint: disable=not-callable
    return tok_tensor


def get_mask_ids(masking):
    if masking == "none":
        mask_ids = []
    elif masking == "random":
        mask_ids = []
    else:
        mask_ids = [int(d) for d in masking.split(',')]
    return mask_ids


def get_seed_sent(seed_sentence, tokenizer, masking, n_append_mask=0):
    """ Get initial sentence to decode from, possible with masks. """

    # Get initial mask
    mask_ids = get_mask_ids(masking)

    # Tokenize, respecting [MASK]
    seed_sentence = seed_sentence.replace(MASK, MASK_ATOM)
    toks = tokenizer.tokenize(seed_sentence)
    toks = trim_tokens(toks, n_append_mask)
    mask_ids = get_mask_ids(masking)
    for i, tok in enumerate(toks):
        if tok == MASK_ATOM:
            mask_ids.append(i)
    org_len = len(toks)
    # Mask the input
    for mask_id in mask_ids:
        toks[mask_id] = MASK

    # Append MASKs
    for _ in range(n_append_mask):
        mask_ids.append(len(toks))
        toks.append(MASK)
    mask_ids = sorted(list(set(mask_ids)))

    seg = [0] * len(toks)
    seg_tensor = torch.tensor([seg]) # pylint: disable=not-callable

    return toks, seg_tensor, mask_ids, org_len


def load_model(version):
    """ Load model. """
    model = BertForMaskedLM.from_pretrained(version)
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda(0)
    return model


def predict(model, tokenizer, tok_tensor, seg_tensor, how_select="argmax"):
    """ Get model predictions and convert back to tokens """
    with torch.no_grad():
        preds = model(tok_tensor, seg_tensor)[0]

    if how_select == "sample":
        dist = Categorical(logits=F.log_softmax(preds[0], dim=-1))
        pred_idxs = dist.sample().tolist()
    elif how_select == "sample_topk":
        raise NotImplementedError("I'm lazy!")
    elif how_select == "argmax":
        pred_idxs = preds.argmax(dim=-1).tolist()[0]
    else:
        raise NotImplementedError("Selection mechanism %s not found!" % how_select)

    pred_toks = tokenizer.convert_ids_to_tokens(pred_idxs)
    return pred_toks


def sequential_decoding(toks, seg_tensor, model, tokenizer, selection_strategy, org_len):
    """ Decode from model one token at a time """
    for step_n in range(len(toks)):
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks, tokenizer)
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor, selection_strategy)
        print("\tBERT prediction: %s" % (" ".join(pred_toks)))
        toks[step_n] = pred_toks[step_n]
    return toks


def masked_decoding(toks, seg_tensor, masks, model, tokenizer, selection_strategy):
    """ Decode from model by replacing masks """
    for step_n, mask_id in enumerate(masks):
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks, tokenizer)
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor, selection_strategy)
        print("\tBERT prediction: %s\n" % (" ".join(pred_toks)))
        toks[mask_id] = pred_toks[mask_id]
    return toks


def generate(version, decoding_strategy, seed_sentence, masking, n_append_mask, token_strategy):
    tokenizer = BertTokenizer.from_pretrained(version)
    model = load_model(version)
    toks, seg_tensor, mask_ids, org_len = get_seed_sent(seed_sentence, tokenizer,
                                               masking=masking,
                                               n_append_mask=n_append_mask)
    text = toks[:org_len]

    if decoding_strategy == "sequential":
        p_toks = sequential_decoding(toks, seg_tensor, model, tokenizer, token_strategy, org_len)
    elif decoding_strategy == "masked":
        p_toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer, token_strategy)
    else:
        raise NotImplementedError("Decoding strategy %s not found!" % decoding_strategy)
    text += (p_toks[org_len:len(p_toks)])

    return text

    # print("Final: %s" % (" ".join(text)))


if __name__ == '__main__':
    # path to model
    # version = "/home/nilo4793/Documents/Bert_Hiwi/transformers/output/10k_run_1/"
    # version = "bert-base-uncased"
    version = "./best_checkpoint_110000_own_vocab/"
    # version = "./best_checkpoint_935000_normal/"
    # choices=["masked", "sequential"]
    decoding_strategy = "sequential"

    # seed_sentence = "I woke up this morning and heard a loud bang."
    seed_sentence = "public static void main(String[] args){ " \
                    "boolean isSet = "
    # seed_sentence = "this is a sentence . "
    # help="Masking strategy: either 'none', 'random', or list of idxs", default="none"
    masking = "none"
    # default=0
    n_append_mask = 2
    # choices=["argmax", "sample", "sample_topk"])
    token_strategy = "sample"

    #pdb.set_trace()
    tokenizer = BertTokenizer.from_pretrained(version)
    model = load_model(version)

    print("Decoding strategy %s, %s at each step" % (decoding_strategy, token_strategy))
    toks, seg_tensor, mask_ids, org_len = get_seed_sent(seed_sentence, tokenizer,
                                               masking=masking,
                                               n_append_mask=n_append_mask)
    print(toks)
    print(len(toks))
    text = toks[:org_len]

    if decoding_strategy == "sequential":
        p_toks = sequential_decoding(toks, seg_tensor, model, tokenizer, token_strategy, org_len)
    elif decoding_strategy == "masked":
        p_toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer, token_strategy)
    else:
        raise NotImplementedError("Decoding strategy %s not found!" % decoding_strategy)
    text += (p_toks[org_len:len(p_toks)])

    print("Final: %s" % (" ".join(text)))
