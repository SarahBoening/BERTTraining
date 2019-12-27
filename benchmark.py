import random
import datetime
import re

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
    model = BertForMaskedLM.from_pretrained(version)
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda(0)
    return model


def predict_token(text, target, model):
    tokenized_text = tokenizer.tokenize(text)
    # print("length: ", len(tokenized_text))

    masked_index = tokenized_text.index(target)
    # print("index: ", masked_index)
    tokenized_text = trim_tokens(tokenized_text, masked_index)
    masked_index = tokenized_text.index(target)
    # print("new length: ", len(tokenized_text))

    # print("new index: ", masked_index)
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [0] * len(tokenized_text)
    token_tensor = torch.tensor([indexed_tokens])
    segment_tensor = torch.tensor([segment_ids])
    # token_tensor = token_tensor.to('cuda')
    # segment_tensor = segment_tensor.to('cuda')
    # model.to('cuda')

    then = datetime.datetime.now()
    with torch.no_grad():
        outputs = model(token_tensor, token_type_ids=segment_tensor)
        predictions = outputs[0]
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    now = datetime.datetime.now()
    print("predicted token: ", predicted_token)
    print("elapsed time for prediction: ", now - then)
    return predicted_token


if __name__ == '__main__':
    # path = '/home/nilo4793/Documents/Bert_Hiwi/corpora/Java_split/0143.java_github_10k.raw'
    path = '0143.java_github_10k.raw'
    content = ''
    f = open(path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    sep = '[SEP]'
    files = [x + sep for x in content.split(sep)]
    # to get rid of trailing
    files = files[:-1]
    random.shuffle(files)
    data = files[:20]
    # further processing for testing
    # add [MASK]
    # tokenize word to see how many [MASK] to add?
    # boolean -> b, ##ool, ##ean -> 3 Masks
    # predict at once or loop?

    # load the model
    modelpath = 'bert-base-cased'
    # modelpath = '/home/nilo4793/Documents/Bert_Hiwi/transformers/output/10k_run_1'
    modelpath = "E:\\PyCharm Projects\\Hiwi\\best_checkpoint_110000_own_vocab"
    # vocab_path = 'bert-base-cased/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    # tokenizer = BertTokenizer.from_pretrained(vocab_path)
    model = load_model(modelpath)
    count = 0
    j = 0
    print("BENCHMARK FOR MODEL ", modelpath.upper())
    for i in range(20):
        no_tokens = 0
        for example in data:
            predicted_word = ''
            target = random.choice(example.split())
            words = re.findall(r'\w+', example)

            target = random.choice(words)
            while len(target) > 10 or target == '[CLS]' or target == '[SEP]' or target == '[MASK]' or '/' in target or '*' in target  or '@' in target:
                target = random.choice(target)
                # target = random.choice(example.split())

            print("target word: ", target)
            example = example.replace(target, '[MASK]', 1)
            tokenized_target = tokenizer.tokenize(target)
            # print(tokenizer.tokenize(target))
            s_target = ""
            for k in range(len(tokenized_target)):
            # for k in range(1):
                s_target = tokenized_target[k]
                print("current target token: ", s_target)

                predicted_token = predict_token(example, '[MASK]', model)
                # predicted_token = tokenized_target[k]
                predicted_word += predicted_token.replace('##', '')
                example = example.replace('[MASK]', predicted_word + ' [MASK]')
                # print(example)

            print("predicted word: ", predicted_word)
            # print("target: ", target)
            if predicted_word == target or predicted_word == s_target:
                count += 1
                print("Correct prediction")
            else:
                print("Wrong prediction")
            print("-------------------------")
            j += 1

    print("No. of made predictions: ", j)
    print("Correct predictions: ", count)
    print("percentage of correct predictions: ", float(count / j) * 100.0, "%")
