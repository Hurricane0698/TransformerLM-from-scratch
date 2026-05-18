'''train_bpe 的输入包括训练文本路径、最终最大 vocab size，以及 special tokens 列表。special tokens 会加入词表，并作为训练时的硬边界，防止 BPE merge 跨越它们，但它们本身不参与 merge 统计。

输出包括：
1. vocab: dict[int, bytes]，从 token ID 映射到 token 的 bytes 表示。
2. merges: list[tuple[bytes, bytes]]，按创建顺序记录每次被合并的两个 token；合并后的新 token 是二者的 bytes 拼接。

BPE merge 只能发生在 pre-token 内部。每一轮统计当前 token 序列里的相邻 pair 频率，选择频率最高的 pair；若并列，则选择字典序更大的 pair；然后把所有该 pair 的出现合并成新 token，加入 vocab，并重复直到达到 vocab_size。'''
import regex as re
from pretokenization_example import find_chunk_boundaries
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def count_chunk(job):
    input_path, start, end, special_tokens= job
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if not special_tokens:
            data_list = [chunk] #[str, str]
    else:
        data_list = re.split("|".join([re.escape(special_token) for special_token in special_tokens]), chunk)
    #counts{ (b'',...):x }
    chunk_counts = Counter()
    #初始化pre-tokenization, count同时计数
    for data in data_list:
        for match in re.finditer(PAT, data): 
            #pre_token_list = re.split(PAT, data)#["a", "Hello", "you-are", ","]
            encoded_pre_token = match.group().encode("utf-8")#b'ab/x0e'
            pre_token_tuple = tuple([bytes([i]) for i in encoded_pre_token])
            chunk_counts[pre_token_tuple] += 1
    return chunk_counts

def train_bpe(input_path:str, vocab_size:int, special_tokens:list) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i : bytes([i]) for i in range(256)}
    #打开路径获得数据集，确定边界，把jobs下发给子进程
    with open(input_path, "rb") as f:
        num_workers = 8
        desired_num_chunks = num_workers * 4
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
    total_counts = Counter()
    jobs = [(input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_counts = executor.map(count_chunk, jobs)
    for partial_count in partial_counts:
        total_counts.update(partial_count)
    counts = total_counts

    merge_list = []
    #初始化pair_counts和pair_to_words
    pair_counts = Counter()#{(b'a', b'c'):14, ...}
    pair_to_words = {}#{(b'c', b'd'):set((b'a', b'b', b'c', b'd')), ...}
    for word_tuple in counts:
        if len(word_tuple) >= 2:
            for i in range(len(word_tuple)-1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_counts[pair] += counts[word_tuple]
                
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_tuple)
    
    #merge循环开始：
    while len(vocab) < vocab_size - len(special_tokens):
        #先确认pair_counts里面仍然有值
        if not pair_counts:
            break
        #贪心取值找best_pair
        best_pair = None
        best_freq = None
        for candidate_pair, candidate_freq in pair_counts.items():
            if best_pair is None:
                best_pair = candidate_pair
                best_freq = candidate_freq
            elif candidate_freq > best_freq:
                best_pair = candidate_pair
                best_freq = candidate_freq
            elif candidate_freq == best_freq and candidate_pair > best_pair:
                best_pair = candidate_pair
        expected = best_pair[0] + best_pair[1] #b'ab'
        #维护merge_list
        merge_list.append(best_pair)
        #加词表
        vocab[len(vocab)] = expected
        #维护三个不变量
        old_words = pair_to_words[best_pair].copy()#set((a,b),(a,b,c), (a,b,a,b)),必须copy，等于是指针，如果直接在下面循环使用会边改边迭代
        for old_word in old_words:
            #先去除pair_counts里旧词的贡献
            for i in range(len(old_word)-1):
                old_pair = (old_word[i], old_word[i+1])
                freq = counts[old_word]
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
            #再维护pair_to_words,按mermbership维护
            unique_old_pairs = set()
            for i in range(len(old_word)-1):
                old_pair = (old_word[i], old_word[i+1])
                unique_old_pairs.add(old_pair)
            for unique_old_pair in unique_old_pairs:
                pair_to_words[unique_old_pair].remove(old_word)
                if not pair_to_words[unique_old_pair]:
                    del pair_to_words[unique_old_pair]

            #先维护counts
            i = 0
            new_word_list = []
            while i <= len(old_word) - 1:
                if i <= len(old_word) - 2:
                    if best_pair == (old_word[i], old_word[i+1]):
                        new_word_list.append(expected)
                        i += 2
                    else:
                        new_word_list.append(old_word[i])
                        i += 1
                else:
                        new_word_list.append(old_word[i])
                        i += 1
            new_word = tuple(new_word_list)
            freq = counts.pop(old_word)
            counts[new_word] += freq
            
            #再维护pair_counts更新
            for i in range(len(new_word)-1):
                new_pair = (new_word[i], new_word[i+1])
                pair_counts[new_pair] += freq
            #再维护pair_to_words更新
            for i in range(len(new_word)-1):
                new_pair = (new_word[i], new_word[i+1])
                if new_pair not in pair_to_words:
                    pair_to_words[new_pair] = set()
                pair_to_words[new_pair].add(new_word)
    #最后加上special_token
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab, merge_list

'''BPE 训练维护三个长期状态：
count: 当前 token tuple -> frequency
pair_counts: 当前相邻 pair -> weighted frequency
pair_to_words: 当前 pair -> 包含它的 token tuple 集合

初始化时，从 pre-tokenization 得到 count，再从 count 建 pair_counts 和 pair_to_words。

每轮 merge：
1. 从 pair_counts 贪心选 best_pair，tie-break 等价于比较 (freq, pair)。
2. 把 best_pair 合成的新 bytes 追加到 vocab，把 best_pair 追加到 merges。
3. 从 pair_to_words[best_pair] 拿 affected words 快照。
4. 对每个 affected old word：
   - 从 pair_to_words 撤销 old word 的 membership。
   - 从 pair_counts 撤销 old word 的 pair 频率贡献。
   - merge old word 得到 new word。
   - 把 new word 的 pair 贡献登记回 pair_counts / pair_to_words。
   - 在 count 中删除 old word，把频率加到 new word。
5. 最后追加 special tokens。
'''
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    input_path = BASE_DIR.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab, merge_list = train_bpe(str(input_path), 10000, ["<|endoftext|>"])

    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merge_list, f)
    longest_token = max(vocab.values(), key=len)
    print(len(longest_token))
    print(longest_token)
    print(longest_token.decode("utf-8", errors="replace"))