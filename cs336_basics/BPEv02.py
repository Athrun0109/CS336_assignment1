'''
在BPEv01.py基础上加入了
1. 多线程预分词
2. pair_freqs采用计数更新，而非重复生成新的pair_freqs
'''

import os
from typing import List, Tuple, Dict, BinaryIO
from collections import defaultdict
import regex as re
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKEN = "<|endoftext|>"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    此处要求file是字节流，而不是字符串流，因此要使用'rb'而不是'r'
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _process_chunk(args) -> Dict[Tuple[int, ...], int]:
    # 作为多线程函数实现预分词
    input_path, start, end, special_tokens = args
    special_tokens = set(special_tokens)
    # start, end对应的是字节偏移量，所以要以'rb'读取，再执行偏移
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore') # 二进制decode为字符串
    # 预分词
    special_pattern = "|".join(re.escape(st) for st in special_tokens) if special_tokens else None
    
    if special_pattern:
        parts = re.split(f"({special_pattern})", chunk)
    else: # chunk中无特殊词
        parts = [chunk]

    word_freqs = defaultdict(int)
    pat = re.compile(PAT)
    for part in parts:
        if part in special_tokens:
            continue
        for match in pat.finditer(part):
            # 这里通过tuple方法将bytes转为ascii码；比如tuple(b'hello') -> (104, 101, 108, 108, 111)
            key = tuple(match.group(0).encode('utf-8'))
            word_freqs[key] += 1

    return word_freqs

def get_pair_freqs(word_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for pre, fol in zip(word[:-1], word[1:]):
            pair_freqs[(pre, fol)] += freq
    
    return pair_freqs

def merge_and_update_freqs(
    word_freqs: Dict[Tuple[int, ...], int],
    pair_freqs: Dict[Tuple[int, int], int],
    pair_to_merge: Tuple[int, int],
    new_id: int
) -> Dict[Tuple[int, ...], int]:
    '''
    此函数为BPEv02中的核心算法，也是最难的部分
    word_freqs在这个函数中主要是更新key
    pair_freqs在这个函数中主要是更新value
    '''
    new_word_freqs = defaultdict(int)
    for word, word_freq in word_freqs.items():
        merged = False
        new_word = []
        i = 0
        while i < len(word):
            cur_pair = word[i:i+2]
            if cur_pair == pair_to_merge:
                merged = True
                if i > 0:
                    prev_ = word[i-1]
                    # 去除旧的配对
                    pair_freqs[(prev_, word[i])] -= word_freq
                    if pair_freqs[(prev_, word[i])] <= 0:
                        pair_freqs.pop((prev_, word[i]))
                    # 增加新的配对
                    pair_freqs[(prev_, new_id)] += word_freq
                if i < len(word) - 2:
                    next_ = word[i+2]
                    # 去除旧的配对
                    pair_freqs[(word[i+1], next_)] -= word_freq
                    if pair_freqs[(word[i+1], next_)] <= 0:
                        pair_freqs.pop((word[i+1], next_))
                    # 增加新的配对
                    pair_freqs[(new_id, next_)] += word_freq
                new_word.append(new_id) # 更新word_freqs的key
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if merged:
            new_word_freqs[tuple(new_word)] += word_freq
        else: # 无事发生的情况，直接获取原来word_freqs的key和value
            new_word_freqs[word] += word_freq

    # 因为pair_to_merge一定被合并了，所以pair_freqs中对应的key和value都要减掉
    pair_freqs.pop(pair_to_merge)

    return new_word_freqs

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    lst = len(special_tokens)
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"Vocab size must be at least 256 + {lst}")
    
    # 多线程实现预分词
    num_processes = os.cpu_count() or 1
    with open(input_path, 'rb') as f:
        # type(boundaries) == List[int]
        boundaries = find_chunk_boundaries(f, num_processes, SPECIAL_TOKEN.encode('utf-8'))
    args = list([input_path, start, end, special_tokens] for start, end in zip(boundaries[:-1], boundaries[1:]))
    with Pool(num_processes) as pool:
        # type(results) == List[Dict[Tuple[int, ...], int]]
        results = pool.map(_process_chunk, args)
    # 合并结果
    word_freqs = defaultdict(int)
    for result in results:
        for key, value in result.items():
            word_freqs[key] += value

    # 初始化返回值vocab以及merges
    vocab = {i: bytes([i]) for i in range(256)} # 注意这里要加[]；并且不是用.encode()
    merges = []
    
    # 初始化一次pair_freqs
    pair_freqs = get_pair_freqs(word_freqs)

    # 开始训练BPE
    num_merges = vocab_size - 256 - lst
    for i in range(num_merges):
        if not pair_freqs:
            print("No more pairs to merge. Stopping early.")
            break
        most_freq_pair = max(pair_freqs, key=lambda x:(pair_freqs[x], x))
        if pair_freqs[most_freq_pair] <= 0: # pair_freqs合并的时候value加的少，减的多，因此value可能为0
            print("Highest frequency pair has count <= 0. Stopping early.")
            break
        new_id = len(vocab)
        word_freqs = merge_and_update_freqs(word_freqs, pair_freqs, most_freq_pair, new_id)
        
        # 更新vocab
        token_bytes1 = vocab[most_freq_pair[0]]
        token_bytes2 = vocab[most_freq_pair[1]]
        vocab[new_id] = token_bytes1 + token_bytes2

        # 更新merges
        merges.append((token_bytes1, token_bytes2))

    # 最后将special_tokens加入vocab
    for st in special_tokens:
        vocab[len(vocab)] = st.encode('utf-8')

    return vocab, merges

if __name__ == "__main__":
    input_file_path = r'C:\Users\Admin\Documents\Codes\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt'
    
    target_vocab_size = 280
    special_tokens = [SPECIAL_TOKEN]
    # pass

    vocab, merges = train_bpe(input_file_path, target_vocab_size, special_tokens)

    print("\n--- BPE Training Finished ---")
    print(f"\nFinal Vocabulary Size: {len(vocab)}")
