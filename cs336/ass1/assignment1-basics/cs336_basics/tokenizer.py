from collections.abc import Iterable, Iterator
import pickle
import regex as re
class Tokenizer:
    def __init__(self, 
                 vocab:dict[int, bytes], 
                 merges:list[tuple[bytes, bytes]], 
                 special_tokens:list[str]|None = None):
        vocab = vocab.copy()
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens.copy()
            self.special_tokens_sorted = sorted(special_tokens.copy(), key=len, reverse=True)
            for special_token in self.special_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in vocab.values():
                    vocab[len(vocab)] = special_token
        self.i2b = vocab
        self.b2i = {b: i for i, b in vocab.items()}
        self.merges = merges.copy()
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    @classmethod
    def from_files(cls, vocab_path:str, merges_path:str, special_tokens:list[str]|None = None):
        with open(vocab_path, "rb") as v:
            vocab = pickle.load(v)
        with open(merges_path, "rb") as m:
            merges = pickle.load(m)
        return cls(vocab, merges, special_tokens)
    def encode(self, text:str) -> list[int]:
        ids = []
        merge_rank = self.merge_rank
        #先处理用户传入的special_tokens，再使用pattern进行tokenization
        if self.special_tokens:
            text_list = re.split("("+"|".join(re.escape(special_token) for special_token in self.special_tokens_sorted)+")", text)
        else:
            text_list = [text]
        for text_part in text_list:
            if text_part == "":
                continue
            elif text_part in self.special_tokens:
                ids.append(self.b2i[text_part.encode("utf-8")])
            else:
                for match in re.finditer(self.pat, text_part):
                    bytes_list = []
                    bytes_list.extend(bytes([i]) for i in match.group().encode("utf-8"))#[b'h', b'e', b'l']
                    while True:    
                        #第一步：贪心找best_pair
                        merge_pair = None
                        best_rank = None
                        for i in range(len(bytes_list)-1):
                            candidate_pair = (bytes_list[i], bytes_list[i+1])
                            if candidate_pair in merge_rank:
                                candidate_rank = merge_rank[candidate_pair]
                                if merge_pair is None:
                                    merge_pair = candidate_pair
                                    best_rank = candidate_rank
                                elif candidate_rank < best_rank:
                                    merge_pair = candidate_pair
                                    best_rank = candidate_rank #这里我前面纯按bpe逻辑套，不行，还是想想条件变没变
                        #第二步：如果有，用mergepair替换list中所有相邻符合项
                        if merge_pair is not None:
                            i = 0
                            while i <= len(bytes_list) - 2:
                                if (bytes_list[i], bytes_list[i+1]) == merge_pair:
                                    bytes_list[i:i+2] = [bytes_list[i] + bytes_list[i+1]] #[b'he', b'l']
                                i += 1
                            continue
                        #第三步，全部完成后退出
                        if not merge_pair:
                            break
                    #每个词完成后把id加到ids末端
                    for finished_bytes in bytes_list:
                        ids.append(self.b2i[finished_bytes])
        return ids 
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        #最重要的思想是保守处理，而不是算法预测所有情况
        for chunk in iterable:
            buffer += chunk
            #对buffer进行special token处理，然后对最后一个内容前面的进行join
            if self.special_tokens:
                buffer_list = re.split("("+"|".join(re.escape(special_token) for special_token in self.special_tokens_sorted)+")", buffer)
                max_special_len = len(max(self.special_tokens, key=len))
            else:
                buffer_list = [buffer]
            if len(buffer_list) >= 2:
                buffer_text = "".join(buffer_list[:-1])
                yield from self.encode(buffer_text)
            buffer = buffer_list[-1] #"wo hai yo,你好，/？<|endof"
            if not buffer:
                continue
            #buffer里有被截断的special_token,得从长到短判断
            held_suffix = None
            #先拿
            if self.special_tokens:
                limit = min(len(buffer), max_special_len - 1)
                for suffix_len in range(limit, 0, -1):#降序实现方法，重点
                    suffix = buffer[-suffix_len:] #反向表示倒数多少个，重点
                    if any(special_token.startswith(suffix) for special_token in self.special_tokens):#any代替正常繁琐循环，重点
                        held_suffix = suffix
                        break
            #再算，同时维护buffer不变量
            if held_suffix is not None:
                safe_text = buffer[:-len(held_suffix)]
                buffer = held_suffix
                yield from self.encode(safe_text)
                continue
            matches = list(re.finditer(self.pat, buffer))
            if len(matches) >= 2:
                safe_end = matches[-1].start() #match的三个方法：group()是内容，start()是原内容的开始索引,end()是结束索引
                safe_text = buffer[:safe_end]
                buffer = buffer[safe_end:]
                yield from self.encode(safe_text) #yield from很好用，直接把可迭代对象内容逐个吐出去
            else:
                continue
        #最后处理buffer剩下片段
        yield from self.encode(buffer)
    def decode(self, ids:list[int]) -> str:
        bytes = b''#先把 bytes 拼起来再 decode。如果是单个 ID 分别 decode 的话，可能会把跨 token 的 UTF-8 多字节字符解释坏
        for token_id in ids:
            bytes += self.i2b[token_id]
        text = bytes.decode(errors="replace")
        return text
    #git add -f强制加入被忽略的内容