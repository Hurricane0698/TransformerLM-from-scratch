
import re


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
allwords = sorted(set(preprocessed))
allwords.extend(["<unk>", "<endoftext>"])
vocab = {t: i for i, t in enumerate(allwords)}


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str2int = vocab
        self.int2str = {i: s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
            ]
        preprocessed = [s if s in self.str2int else "<unk>" for s in preprocessed]
        ids = [self.str2int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int2str[i] for i in ids])
        text = re.sub(r'\s([,.?_!"()\']|--)\s', r'\1', text)
        return text
 
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = "Akwirw ier"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))