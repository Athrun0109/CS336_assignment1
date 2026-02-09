"""
只需运行一次；训练BPE Tokenizer并将Text -> IDs
"""
import os
import json
import math
import argparse

import numpy as np
import regex as re
from tqdm import tqdm

import BPEv02

SPECIAL_TOKENS = [BPEv02.SPECIAL_TOKEN]


def save_tokenizer(path, vocab, merges, train_data_path, vocab_size, special_tokens):
    # 将 BPE 训练结果保存为 JSON（bytes 用 hex 编码）
    data = {
        "train_data_path": train_data_path,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "vocab": {str(k): v.hex() for k, v in vocab.items()},
        "merges": [[a.hex(), b.hex()] for a, b in merges],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Tokenizer saved to {path}")


def load_tokenizer(path):
    # 从 JSON 加载 BPE 训练结果，返回 (vocab, merges, meta)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
    meta = {
        "train_data_path": data["train_data_path"],
        "vocab_size": data["vocab_size"],
        "special_tokens": data["special_tokens"],
    }
    print(f"Tokenizer loaded from {path}")
    return vocab, merges, meta


class BPETokenizer:
    def __init__(self, vocab, merges, patterns):
        self.vocab = vocab
        self.merges = merges
        self.patterns = re.compile(patterns)

        self.vocab_inverse = {v: k for k, v in vocab.items()}
        self.merges_rank = {v: i for i, v in enumerate(merges)}

    def _bpe_merge(self, tbl):
        """
        Bytes -> IDs
        Args:
            List[Bytes]
        Returns:
            List[int]
        """
        while len(tbl) >= 2:
            min_rank = math.inf
            best_pair = None

            for i in range(len(tbl)):
                if i < len(tbl) - 1 and (tbl[i], tbl[i+1]) in self.merges_rank:
                    rank = self.merges_rank[(tbl[i], tbl[i+1])]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = (tbl[i], tbl[i+1])

            # 没有可合并的内容，直接结束循环
            if best_pair is None:
                break

            # 合并best_pair
            new_tbl = []
            i = 0
            while i < len(tbl):
                if i < len(tbl) - 1 and (tbl[i], tbl[i+1]) == best_pair:
                    new_tbl.append(tbl[i] + tbl[i+1])
                    i += 2
                else:
                    new_tbl.append(tbl[i])
                    i += 1

            tbl = new_tbl

        # Bytes -> IDs
        result = []
        for b in tbl:
            if b in self.vocab_inverse:
                result.append(self.vocab_inverse[b])
            else:
                print(f"Warning: unknown token '{b}'")

        return result

    def encode(self, input_file_path, desc="Encoding"):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        result = []
        # 先将特殊字符与其他内容分割开
        special_pattern = "|".join(re.escape(st) for st in SPECIAL_TOKENS)
        chunks = re.split(f'({special_pattern})', text) # 注意这个()，用来保留特殊字符串

        # 编码
        for chunk in tqdm(chunks, desc=desc):
            # 特殊字符直接转为IDs
            if chunk in SPECIAL_TOKENS:
                idx = self.vocab_inverse.get(bytes(chunk, 'utf-8'), None)
                if idx is not None:
                    result.append(idx)
                else:
                    print(f"Warning: Unknown special token '{chunk}'")
            else:
                # 使用patterns分词
                text_chunks = self.patterns.findall(chunk)
                # 将字符串转为字节形式
                for text_chunk in text_chunks:
                    token_bytes_list = [bytes([b]) for b in text_chunk.encode('utf-8')]
                    encoded_ids = self._bpe_merge(token_bytes_list)
                    result.extend(encoded_ids)

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r'C:\Users\Admin\Documents\Codes\assignment1-basics')
    parser.add_argument("--train_input", type=str, default=r"data\TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_input", type=str, default=r"data\TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer JSON. If exists and params match, skip BPE training.")
    args = parser.parse_args()

    train_output_path = os.path.join(args.root_path, r"data\train.npy")
    val_output_path = os.path.join(args.root_path, r"data\val.npy")

    train_data_path = os.path.join(args.root_path, args.train_input)
    val_data_path = os.path.join(args.root_path, args.val_input)

    # 默认 tokenizer 保存路径
    tokenizer_path = args.tokenizer_path or os.path.join(args.root_path, "data", "tokenizer.json")

    # 1. 训练 BPE 或从缓存加载
    loaded = False
    if os.path.exists(tokenizer_path):
        vocab, merges, meta = load_tokenizer(tokenizer_path)
        # 校验关键参数是否匹配
        if (meta["train_data_path"] == train_data_path
                and meta["vocab_size"] == args.vocab_size
                and meta["special_tokens"] == SPECIAL_TOKENS):
            print("Tokenizer params match, skipping BPE training.")
            loaded = True
        else:
            print("Tokenizer params mismatch, re-training...")

    if not loaded:
        print(f"Training BPE vocab (size={args.vocab_size})...")
        vocab, merges = BPEv02.train_bpe(train_data_path, args.vocab_size, SPECIAL_TOKENS)
        save_tokenizer(tokenizer_path, vocab, merges, train_data_path, args.vocab_size, SPECIAL_TOKENS)

    tokenizer = BPETokenizer(vocab, merges, BPEv02.PAT)

    # 2. 处理并保存训练集
    print(f"Processing {train_data_path}...")
    train_token_IDs = tokenizer.encode(train_data_path, desc="Encoding train")
    # 使用 uint16 节省空间 (前提是 vocab_size < 65536)
    np.save(train_output_path, np.array(train_token_IDs, dtype=np.uint16))
    print(f"Saved to {train_output_path}")

    # 3. 处理并保存验证集
    print(f"Processing {val_data_path}...")
    val_token_IDs = tokenizer.encode(val_data_path, desc="Encoding val")
    np.save(val_output_path, np.array(val_token_IDs, dtype=np.uint16))
    print(f"Saved to {val_output_path}")