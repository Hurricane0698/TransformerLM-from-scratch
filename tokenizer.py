'''SimpleTokenizer

状态：
- str2int: token -> id
- int2str: id -> token

encode(text):
1. 用正则把文本切成 token
2. 去掉空白 token
3. 如果 token 在 vocab 里，用对应 id
4. 如果不在，用 unk id
5. 返回 id 列表

decode(ids):
1. 把 id 转回 token
2. 用空格 join
3. 删除标点符号前多余空格
4. 返回文本'''
import re
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str2int = vocab
        self.int2str = {i : s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!()"\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str2int[token] if token in self.str2int else self.str2int["<unk>"]
                for token in preprocessed]
    def decode(self, ids):
        text = " ".join(self.int2str[token_id] for token_id in ids)
        text = re.sub(r'\s+([,.:;?_!()"\']|--)', r'\1', text)
        return text

def run_tokenizer_tests(TokenizerClass):
    # 1. 构造一个小词表，专门用于测试
    vocab_tokens = [
        "Hello", ",", "world", "!", 
        "Do", "you", "like", "tea", "?",
        "I", "am", "fine", ".",
        "--", 
        "<|endoftext|>", "<|unk|>"
    ]

    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    tokenizer = TokenizerClass(vocab)

    def ids(tokens):
        return [vocab[t] for t in tokens]

    # 2. 基础标点分词测试
    assert tokenizer.encode("Hello, world!") == ids(["Hello", ",", "world", "!"])

    # 3. decode 是否能去掉标点前多余空格
    assert tokenizer.decode(ids(["Hello", ",", "world", "!"])) == "Hello, world!"

    # 4. 问号、句号测试
    assert tokenizer.encode("Do you like tea?") == ids(["Do", "you", "like", "tea", "?"])
    assert tokenizer.decode(ids(["Do", "you", "like", "tea", "?"])) == "Do you like tea?"

    # 5. 未知词测试
    assert tokenizer.encode("Hello AI!") == ids(["Hello", "<|unk|>", "!"])

    # 6. <|endoftext|> 特殊 token 测试
    assert tokenizer.encode("Hello <|endoftext|> world!") == ids(
        ["Hello", "<|endoftext|>", "world", "!"]
    )

    # 7. 双破折号测试
    assert tokenizer.encode("Hello--world!") == ids(["Hello", "--", "world", "!"])

    # 8. encode-decode-encode 闭环测试
    original_ids = ids(["I", "am", "fine", "."])
    decoded = tokenizer.decode(original_ids)
    encoded_again = tokenizer.encode(decoded)
    assert encoded_again == original_ids

    print("All tokenizer tests passed.")
