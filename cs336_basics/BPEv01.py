import os
from typing import List, Tuple, Dict
from collections import defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

test_text = '''
    He leaned against her and she kept him safe. The mole had found his best friend.
    <|endoftext|>
    Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
'''

def get_pair_freqs(word_freqs:Dict[Tuple[int], int]) -> Dict[Tuple[int, int], int]:
    '''
    Inputs:
        word_freqs: dict[Tuple[int], int] A dictionary mapping int (word ID) to int (word frequency).
    Returns:
        pair_freqs: dict[tuple[int, int], int] A dictionary mapping tuples of int (word pair ID) to int (pair frequency).
    '''
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for pre, fol in zip(word[:-1], word[1:]):
            pair_freqs[(pre, fol)] += freq

    return pair_freqs

def merge_word_freqs(word_freqs:Dict[Tuple[int], int], most_frequent_pair:Tuple[int, int], new_token_id:int) -> Dict[Tuple[int], int]:
    '''
    Inputs:
        word_freqs: dict[Tuple[int], int] A dictionary mapping int (word ID) to int (word frequency).
        most_frequent_pair: tuple[int, int] A tuple of int (word pair ID) to merge.
        new_token_id: int The ID of the new token that will be created by merging the target pair.
    Returns:
        new_word_freqs: dict[Tuple[int], int] A dictionary mapping int (word ID) to int (word frequency) after merging.
    '''
    new_word_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        i = 0
        new_key= []
        while i < len(word):
            if i < len(word) - 1 and word[i:i+2] == most_frequent_pair:
                new_key.append(new_token_id)
                i += 2
            else:
                new_key.append(word[i])
                i += 1
        new_word_freqs[tuple(new_key)] += freq

    return new_word_freqs

def train_bpe(input_path:str, vocab_size:int, special_tokens:List[str]):
    '''
    Inputs:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
                initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
                        otherwise affect BPE training.
    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
                is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
                <token2>. The merges should be ordered by order of creation.
    '''
    lst = len(special_tokens)
    if vocab_size < 256 + lst:
        raise ValueError(f"vocab_size must be at least 256 + {lst}")
    special_tokens_set = set(special_tokens)
    vocab = {}
    merges = []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. 初始化前256个byte token
    for i in range(256):
        vocab[i] = bytes([i]) # 注意这里是[i]而不是i，这和bytes的定义有关。
    
    # 2. 将文本根据special_tokens分割为多个chunk
    special_pattern = "|".join(re.escape(st) for st in special_tokens)
    text_chunks = re.split(f'({special_pattern})', text)
    # print(text_chunks)

    # 3. 遍历text_chunks，根据PAT进行分词，并计算词频
    word_freqs = defaultdict(int)
    pre_tokenizer = re.compile(PAT)
    for i, chunk in enumerate(text_chunks):
        if chunk in special_tokens_set: # 跳过特殊token
            continue
        for match in pre_tokenizer.finditer(chunk):
            b = match.group(0).encode("utf-8")
            word_freqs[tuple(b)] += 1 # 这里通过tuple方法将bytes转为ascii码；比如tuple(b'hello') -> (104, 101, 108, 108, 111)

    # 4. 构建训练BPE的循环
    num_merges = vocab_size - lst - 256
    for i in range(num_merges):
        # 5. 遍历词频字典，然后计算出前后字母组合的频率
        pair_freqs = get_pair_freqs(word_freqs)

        # 6. 找出出现次数最多的前后字母组合，以字典序大的优先
        most_frequent_pair = max(pair_freqs, key=lambda k:(pair_freqs[k], k)) # max(dict)返回最大的key值

        # 7. 合并词频字典中出现次数最多的前后字母组合，并更新词频字典
        new_token_id = 256 + i
        word_freqs = merge_word_freqs(word_freqs, most_frequent_pair, new_token_id)

        # 8. 更新vocab字典; vocab: dict[int, bytes]
        token1_bytes = vocab[most_frequent_pair[0]]
        token2_bytes = vocab[most_frequent_pair[1]]
        vocab[new_token_id] = token1_bytes + token2_bytes

        # 9. 更新merges列表; merges: list[tuple[bytes, bytes]]
        merges.append((token1_bytes, token2_bytes))

        # 打印详细信息
        # print(f"Merge {i+1}/{num_merges}: {most_frequent_pair} -> {new_token_id} (Freq: {pair_freqs[most_frequent_pair]})")

    # 10. 在vocab最后添加上special_tokens
    for st in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = st.encode("utf-8")

    return vocab, merges


if __name__ == '__main__':
    # training_text = (
    #     "Hello world!\n"
    #     "This is a simple test for BPE.\n"
    #     "Repetition is the key, repetition is important.\n"
    #     "simple simple simple test."
    # )
    # input_file_path = "train_data.txt"
    # with open(input_file_path, "w", encoding="utf-8") as f:
    #     f.write(training_text)
    # ret = re.findall(PAT, test_text)
    # print(ret)

    input_file_path = r'C:\Users\Admin\Documents\Codes\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt'
    
    target_vocab_size = 280
    special_tokens_list = ["<|endoftext|>", "<|pad|>"]

    special_pattern = "|".join(re.escape(st) for st in special_tokens_list)
    text_chunks = re.split(f'({special_pattern})', test_text) # 使用捕获组保留分隔符
    print(text_chunks)

    pre_tokenizer = re.compile(PAT)
    for match in pre_tokenizer.finditer(test_text):
        print(match.group(0))
        print(match.group(0).encode('utf-8')) # 这里还是b...形式的bytes
        print(tuple(match.group(0).encode('utf-8'))) # 通过tuple方法能将bytes转为ascii码；比如tuple(b'hello') -> (104, 101, 108, 108, 111)
    
    print("Starting OPTIMIZED BPE training...")
    
    # 调用优化后的新函数
    trained_vocab, trained_merges = train_bpe(
        input_path=input_file_path,
        vocab_size=target_vocab_size,
        special_tokens=special_tokens_list
    )
    
    print("\n--- BPE Training Finished ---")
    print(f"\nFinal Vocabulary Size: {len(trained_vocab)}")