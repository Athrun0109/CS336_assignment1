'''
生成代码
'''
import regex as re
from typing import List, Tuple

import torch
import torch.nn.functional as F

import BPEv02
from modules import TransformerLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BPETokenizer:
    def __init__(self, vocab, merges, patterns):
        self.vocab = vocab # {int: bytes}
        self.merges = merges
        self.patterns = re.compile(patterns)

        self.merges_rank = {v: i for i, v in enumerate(merges)}
        self.vocab_inverse = {v: i for i, v in vocab.items()}

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
            while i < len(token_bytes_list):
                if i < len(token_bytes_list) - 1 and (token_bytes_list[i], token_bytes_list[i + 1]) == best_pair:
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
                if token_bytes in self.vocab_inverse:
                    ids.append(self.vocab_inverse[token_bytes])
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
        combined_bytes = b''.join(bytes_list) # 这里的b''与''.encode('utf-8')、bytes()效果相同
        # 3. 解析为字符串
        decoded_str = combined_bytes.decode('utf-8', errors='replace')

        return decoded_str

def top_p_sampling(logits, p=0.9):
    """
    logits.shape = (1, vocab_size)
    Description:
        top_p_sampling算法原理是先对logits从大到小排序，然后获取前面的累积概率和恰巧超过p的元素，将这部分概率值从新归一化到和为1.0。
    """
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

@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_new_tokens, context_length, temperature, top_p, device):
    model.eval()

    # Encode prompt (text -> ids)
    input_ids = tokenizer.encode(prompt_text)
    ## 转为torch.tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)

    # 获取结束符id
    end_bytes = bytes(BPEv02.SPECIAL_TOKEN, encoding="utf-8")
    eos_id = tokenizer.vocab_inverse.get(end_bytes, None)

    # 生成新字符的循环
    for _ in range(max_new_tokens):
        # 截取input_tensor最后context_length长度
        input_tensor_slice = input_tensor[:, -context_length:]

        # 模型预测/前传
        logits = model(input_tensor_slice)
        logits_last = logits[:, -1, :] # shape=(1, vocab_size)

        # 采样(top_p_sampling、argmax...)
        if temperature == 0.0:
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
        elif temperature > 0.0:
            logits_last = logits_last / temperature
            if 0 < top_p < 1:
                next_token = top_p_sampling(logits_last, p=top_p)
            else:
                probs = torch.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            raise ValueError("Temperature must be ≥ 0.0!")

        # 将新的token拼接到input_tensor后面
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

        # 检查结束符eos_id
        if eos_id is not None and next_token.item() == eos_id:
            print(f"Hit {BPEv02.SPECIAL_TOKEN}, stopping.")
            break

    # Decode result (ids -> text)
    generated_text = tokenizer.decode(input_tensor[0].tolist())

    return generated_text

if __name__ == "__main__":
    input_file_path = r'C:\Users\Admin\Documents\Codes\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt'
    prompt = "Once upon a time"

    vocab_size = 300
    # 训练bpe分词，获取vocab和merges
    vocab, merges = BPEv02.train_bpe(input_file_path, vocab_size, [BPEv02.SPECIAL_TOKEN])

    # 初始化推理用的 Tokenizer
    tokenizer = BPETokenizer(vocab, merges, BPEv02.PAT)

    # 实例化推理模型
    model = TransformerLM(
        vocab_size=len(vocab),
        max_seq_len=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32
    ).to(device)

    print(f"\nGenerating from prompt: {prompt}")

    # 生成文本
    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=50,
        context_length=64,
        temperature=0.8,
        top_p=0.9,
        device=device
    )

    print('-' * 20)
    print("Result:")
    print(result)