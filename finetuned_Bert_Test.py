import torch
from transformers import BertTokenizer, BertForMaskedLM

modelpath = "/home/nilo4793/Documents/Bert_Hiwi/transformers/output/10k_run_1/"
tokenizer = BertTokenizer.from_pretrained(modelpath)
# modified Java vocab
# tokenizer = BertTokenizer.from_pretrained('/home/nilo4793/Documents/Bert_Hiwi/transformers/bert-base-cased/vocab.txt')

# text = "public static void main(String[] args) { target"
# output: void
# text = "public static void main ()"

# text = "public static void main(String[] args) { long foo = 10000; int t target }"
# text = "abstract assert boolean break byte case catch char class continue default do double else enum exports extends final finally float for if implements import instanceof int interface long module native new package private protected public requires return short static strictfp  super switch synchronized  this throw throws transient try void volatile while  true null false ; , . ( ) [ ] { }  =  == // /* */ /**  **/  L D"
# output:

text = "public boolean removeAngle(double angle) { " \
       "    for (Iterator i = angles.iterator(); i.hasNext(); ) { " \
       "       Double r = (Double)i.next(); " \
       "       if (r.doubleValue() == angle) { " \
       "           return removeAngle(r); " \
       "       } " \
       "   } " \
       "   return false; " \
       "}"

text = "protected boolean hasModelChildrenChanged(Notification evt) {" \
       "   return false;" \
       "}"
# text = "public static void main ()"
# text = "public static void main(String[] args) { long foo = 10000; int t target }"

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

target = "boolean"
masked_index = tokenized_text.index(target)
# masked_index = 8
tokenized_text[masked_index] = '[MASK]'

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0] * len(tokenized_text)
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(modelpath)
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print("Original:", text)
print("Masked:", " ".join(tokenized_text))

print("Predicted token:", predicted_token)
print("Other options:")
for i in range(10):
    predictions[0, masked_index, predicted_index] = -11100000
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    print(predicted_token)
