import torch
from transformers import BertTokenizer, BertForMaskedLM
from bert_generation import generate


def trim_tokens(tokenized_text, masked_index):
    if len(tokenized_text) > 512:
        if masked_index < 511:
            tokenized_text = tokenized_text[0:511]
        elif masked_index > len(tokenized_text) - 512:
            tokenized_text = tokenized_text[-512:]
        else:
            tokenized_text = tokenized_text[(masked_index - 256):(masked_index + 256)]

    return tokenized_text


def load_model(version):
    """ Load model. """
    print("loading model...")
    model = BertForMaskedLM.from_pretrained(version)
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda(0)
    return model


if __name__ == '__main__':
    # path = '/home/nilo4793/Documents/Bert_Hiwi/corpora/Java_split/0143.java_github_10k.raw'
    path = 'E:\\PyCharm Projects\\Hiwi\\Grill.java'
    content = ''
    f = open(path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    # load the model
    modelpath = 'bert-base-cased'
    # modelpath = '/home/nilo4793/Documents/Bert_Hiwi/transformers/output/10k_run_1'
    modelpath = "E:\\PyCharm Projects\\Hiwi\\best_checkpoint-1820000_multi"
    # vocab_path = 'bert-base-cased/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    # tokenizer = BertTokenizer.from_pretrained(vocab_path)
    #model = load_model(modelpath)
    count = 0
    j = 0
    print("GENERATION TEST FOR MODEL ", modelpath.upper())
    no_tokens = 10

    #print("SEQUENTIAL, NO MASKING SAMPLING")
    #text_pred = generate(modelpath, "sequential", content, "none", no_tokens, "sample")
    #print("Final: %s" % (" ".join(text_pred)))
    #print("-------------------------")
    print("MASKED, NO MASKING, SAMPLING")
    text_pred = generate(modelpath, "masked", content, "none", no_tokens, "sample")
    print("Final: %s" % (" ".join(text_pred)))
    print("-------------------------")
