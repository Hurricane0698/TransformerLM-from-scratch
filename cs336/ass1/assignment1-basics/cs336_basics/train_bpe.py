'''train_bpe 的输入包括训练文本路径、最终最大 vocab size，以及 special tokens 列表。special tokens 会加入词表，并作为训练时的硬边界，防止 BPE merge 跨越它们，但它们本身不参与 merge 统计。

输出包括：
1. vocab: dict[int, bytes]，从 token ID 映射到 token 的 bytes 表示。
2. merges: list[tuple[bytes, bytes]]，按创建顺序记录每次被合并的两个 token；合并后的新 token 是二者的 bytes 拼接。

BPE merge 只能发生在 pre-token 内部。每一轮统计当前 token 序列里的相邻 pair 频率，选择频率最高的 pair；若并列，则选择字典序更大的 pair；然后把所有该 pair 的出现合并成新 token，加入 vocab，并重复直到达到 vocab_size。'''
import regex as re
def train_bpe(input_path:str, 
              vocab_size:int, 
              special_tokens:list) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    with open(input_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    #按空和非空list逻辑处理
    if special_tokens:
        #先处理<|endoftext|>里的特殊元素防止分错，再按re.split|逻辑来分
        train_data_list = re.split(("|".join([re.escape(token) for token in special_tokens])), 
                            train_data)
    else:
        train_data_list = [train_data]
    #pre-tokenization
    count = {} #count要对全数据集计数，学习整个训练数据集的统计分布。
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for data in train_data_list:
        #encode，计数
        for match in re.finditer(PAT, data):
            pre_token = match.group().encode("utf-8") #eg:b'a'
            pre_token_list = []
            for i in pre_token:
                pre_token_list.append(bytes([i]))
            pre_token_tuple = tuple(pre_token_list)
            #先检查是否存在
            if pre_token_tuple in count:
                count[pre_token_tuple] += 1
            else:
                count[pre_token_tuple] = 1
    merge_list = []
    while len(vocab) < vocab_size-len(special_tokens):
        pair_counts = {}
        #统计次数比先统计长度判断是否还有两个以上元素更好，因为没有次数就没有两个以上的元素，最终落点是在次数判断上。
        for per_token_tuple in count:
            if len(per_token_tuple) >= 2:#先判断元组不是单元素
                for i in range(0, len(per_token_tuple)-1):
                        sub_tuple = tuple([per_token_tuple[i], per_token_tuple[i+1]])
                        if sub_tuple in pair_counts:
                            pair_counts[sub_tuple] += count[per_token_tuple] #加每个token tuple的计数，这里之前看错变量了
                        else:
                            pair_counts[sub_tuple] = count[per_token_tuple]
            else:
                continue
        if not pair_counts:#if pair_counts is None 判断不对,pair_counts 初始化为 {}，没有 pair 时它是空 dict，不是 None。
            break
        max_counts = max(pair_counts.values())
        #首先判断有没有多个，有多个的话取字表靠后的（也就是id更高的），否则就选择最高的合并
        max_tup_list = []
        for _, (tup, freq) in enumerate(pair_counts.items()):
            if freq == max_counts:
                max_tup_list.append(tup)
        if len(max_tup_list) > 1:
            #先把每个tuple拿出来，然后比较第一个位置第一个byte数值，不行就第二个，直到元组第一个位置遍历完转到第二个，重复比较
        #若为1，应该是把 merge 的那两个元素拿出来合并成一个，然后索引加在 vocab 的里面，同时 merge list 也加
            best_pair = max(max_tup_list)
            expected = best_pair[0] + best_pair[1] #得到的就是list，不用继续索引了
        else:
            best_pair = max_tup_list[0]
            expected = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = expected
        merge_list.append(best_pair)
        new_count = {}
        for per_token_tuple2 in count:
            if len(per_token_tuple2) >= 2:
                new_per_token_list = []
                i = 0
                while i <= len(per_token_tuple2)-1:
                    if i <= len(per_token_tuple2) - 2:
                        if best_pair == tuple([per_token_tuple2[i], per_token_tuple2[i+1]]):
                            new_per_token_list.append(expected)
                            i += 2
                        else:
                            new_per_token_list.append(per_token_tuple2[i])
                            i += 1
                    elif i == len(per_token_tuple2)-1:
                        new_per_token_list.append(per_token_tuple2[i])
                        i += 1
                new_per_token_tuple = tuple(new_per_token_list)
            else:
                new_per_token_tuple = per_token_tuple2
            if new_per_token_tuple in new_count:
                new_count[new_per_token_tuple] += count[per_token_tuple2]
            else:
                new_count[new_per_token_tuple] = count[per_token_tuple2]
        count = new_count
                            
    #最后加上特殊符号
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")#str->bytes
    return vocab, merge_list