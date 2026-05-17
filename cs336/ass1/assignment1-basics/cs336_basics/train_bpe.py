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
    count = {} 
    ''' count:
            token_tuple -> frequency
        pair_counts:
            pair -> total_frequency
        pair_to_words:
            pair -> set of token_tuple that currently contain this pair'''
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #count初始化
    for data in train_data_list:
        #encode，计数
        for match in re.finditer(PAT, data):
            pre_token = match.group().encode("utf-8") #eg:b'a'
            pre_token_list = []
            for i in pre_token:
                pre_token_list.append(bytes([i]))
            pre_token_tuple = tuple(pre_token_list)
            if pre_token_tuple in count:
                count[pre_token_tuple] += 1
            else:
                count[pre_token_tuple] = 1

    merge_list = []
    pair_counts = {} #{(b'a', b'b'):x}
    pair_to_words = {}#{(b'a', b'b'):set((b'a', b'b', b'c'),(b'g', b'a', b'b'),....)}

    #pair_counts和pair_to_words初始化
    for per_token_tuple in count: #(b'a', b'c', b'd')
        if len(per_token_tuple) >= 2:#先判断元组不是单元素
            for i in range(0, len(per_token_tuple)-1):
                sub_tuple = (per_token_tuple[i], per_token_tuple[i+1])
                if sub_tuple not in pair_to_words:
                    pair_to_words[sub_tuple] = set()
                pair_to_words[sub_tuple].add(per_token_tuple)
                if sub_tuple in pair_counts:
                    pair_counts[sub_tuple] += count[per_token_tuple]
                else:
                    pair_counts[sub_tuple] = count[per_token_tuple] 
        else:
            continue

    #merge best pair, 更新count值、pair_counts、pair_to_words三个全局状态
    while len(vocab) < vocab_size-len(special_tokens):
        #先判断有pair可选
        if not pair_counts:
            break
        #贪心更新
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

        #词汇表增加merge后的bytes
        expected = best_pair[0] + best_pair[1] #b'ab'
        vocab[len(vocab)] = expected
        merge_list.append(best_pair)
        
        #对于影响的word，定向merge best_pair
        affected_words = pair_to_words[best_pair].copy()#set((b'a', b'b', b'c'),(b'g', b'a', b'b'),....),用copy是因为=是指针，copy才是副本，迭代不会出问题
        for affected_word in affected_words:
            i = 0 #维护扫描更新merge_word的指针

            #pair_to_words 只维护 membership，所以同一个 old word 对同一个 pair 只 remove 一次
            old_pairs = set()
            for p in range(len(affected_word)-1):
                key = (affected_word[p], affected_word[p+1])
                old_pairs.add(key)
            for k in old_pairs:
                pair_to_words[k].remove(affected_word)
                if not pair_to_words[k]:
                    del pair_to_words[k]
            
            #先读贡献，再撤销pair在pair_counts里的贡献
            frequency = count[affected_word]
            new_per_token_list = []
            for c in range(len(affected_word)-1):
                pair = (affected_word[c], affected_word[c+1])
                pair_counts[pair] -= frequency
                if pair_counts[pair] == 0:#如果值已经清零就删掉
                    del pair_counts[pair]
            
            #把含best_pair的词更新        
            while i <= len(affected_word)-1:
                if i <= len(affected_word) - 2:
                    if best_pair == (affected_word[i], affected_word[i+1]):
                        new_per_token_list.append(expected)
                        i += 2
                    else:
                        new_per_token_list.append(affected_word[i])
                        i += 1
                elif i == len(affected_word)-1:
                    new_per_token_list.append(affected_word[i])
                    i += 1
            new_per_token_tuple = tuple(new_per_token_list)
            
            #使用新的new_per_token_tuple计算新的贡献并更新pair_counts、pair_to_words
            for j in range(len(new_per_token_tuple)-1):
                    new_pair = (new_per_token_tuple[j], new_per_token_tuple[j+1])#(b'ab', b'c')
                    #更新pair_to_word映射
                    if new_pair not in pair_to_words:
                        pair_to_words[new_pair] = set()
                    pair_to_words[new_pair].add(new_per_token_tuple)
                    #更新pair_counts映射
                    if new_pair in pair_counts:
                        pair_counts[new_pair] += frequency
                    else:
                        pair_counts[new_pair] = frequency
            freq = count.pop(affected_word)
            
            #更新count
            if new_per_token_tuple in count:
                count[new_per_token_tuple] += freq
            else:
                count[new_per_token_tuple] = freq
    #词表最后加上特殊符号
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")#str->bytes
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