'''
生成代码
'''
import os
import time
import regex as re
from typing import List, Tuple

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import BPEv02
from modules import (get_batch, TransformerLM, AdamW, SGD, cross_entropy_loss, gradient_clipping, save_checkpoint, \
                     load_checkpoint, cosine_annealing)

class BPETokenizer:
    def __init__(self, vocab, merges, patterns):
        self.vocab = vocab # {int: bytes}
        self.merges = merges
        self.patterns = re.compile(patterns)

        self.merges_rank = {v: i for i, v in enumerate(merges)}
        self.reversed_vocab = {v: i for i, v in vocab.items()}

    def _bpe_merge(self, token_bytes_list):
        """
        Args:
            token_bytes_list: 元素为bytes的列表
        Returns:
            token_bytes_list: 相对输入，返回的list部分元素已合并，即长度缩短
        要注意的一点是每次循环进行merge的时候，可能存在多个需要merge的前后对！
        """
        while len(token_bytes_list) >= 2:
            min_rank = float('inf')
            best_pair = None
            for i, couple in enumerate(zip(token_bytes_list[:-1], token_bytes_list[1:])):
                rank = self.merges_rank.get(couple, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = couple # tuple
            if best_pair is None:
                break
            pair_to_merge = best_pair[0] + best_pair[1]
            new_token_bytes_list = []
            i = 0
            while i < len(token_bytes_list) - 1:
                if (token_bytes_list[i], token_bytes_list[i + 1]) == best_pair:
                    new_token_bytes_list.append(pair_to_merge)
                    i += 2
                else:
                    new_token_bytes_list.append(token_bytes_list[i])
                    i += 1
            token_bytes_list = new_token_bytes_list

        return token_bytes_list

    def encode(self, text):
        """
        String -> List[int]
        """
        ids = []
        # 1. 使用regex和PAT进行分词
        text_chunks = self.patterns.findall(text)

        # 2. 将text_chunks转为List[bytes]
        for text_chunk in text_chunks:
            token_bytes_list = [bytes([b]) for b in text_chunk.encode('utf-8')]
            merged_bytes = self._bpe_merge(token_bytes_list)

            # 3. 查表将token_bytes转为int
            for token_bytes in merged_bytes:
                if token_bytes in self.reversed_vocab:
                    ids.append(self.reversed_vocab[token_bytes])
                else:
                    print(f"Warning: Unknown token: {token_bytes}")

        return ids

    def decode(self, ids: List[int]):
        """
        List[int] -> String
        """
        bytes_list = []
        # 1. ID -> bytes_list
        for idx in ids:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            else:
                print(f"Warning: Unknown idx: {idx}")

        # 2. 将list合并
        combined_bytes = b''.join(bytes_list)
        # 3. 解析为字符串
        decoded_str = combined_bytes.decode('utf-8', errors='replace')

        return decoded_str

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    logits.shape = (1, vocab_size)
    """
    if temperature > 0:
        logits = logits / temperature

    # top_p_sampling算法原理是先对logits从大到小排序，然后获取前面的累积概率和恰巧超过p的元素，将这部分概率值从新归一化到和为1.0。
    # 1. 对第1维从大到小进行排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # 2. 将概率值归一化
    sorted_probs = F.softmax(sorted_logits, dim=1)
    # 3. 获取需要被移除的元素下标
    cum_probs = torch.cumsum(sorted_probs, dim=1)
    sorted_indices_to_remove = cum_probs > p
    # 3.1. 保留第一个大于p的元素
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    # 3.2. 通过Tensor.scatter将需要被移除的下标分散/分发回排序前的位置
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    # 4. 将需要被移除的元素设置为-inf
    logits[indices_to_remove] = float('-inf')
    # 5. 重新归一化
    probs = F.softmax(logits, dim=1)
    # 6. 重新采样
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token

if __name__ == "__main__":
    input_file_path = r'C:\Users\Admin\Documents\Codes\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt'

    vocab_size = 300
    vocab, merges = BPEv02.train_bpe(input_file_path, vocab_size, BPEv02.SPECIAL_TOKENS)

    pass