import os
import torch
import datasets
from datasets import load_dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset
import itertools

data_path = '/chenyaofo/SlimPajama-627B'
teacher_path = "/chenyaofo/hf_models/DeepSeek-V2-Lite-Chat"
dst_path = f'/chenyaofo/datasets/slimpajama-dsv2-distill/parquet_files2'

class TakeIterableDataset(IterableDataset):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples

    def __iter__(self):
        yield from itertools.islice(self.dataset, self.num_samples)

    def __len__(self):
        return self.num_samples

def prepare_slimpajamar_dataset_for_distillation(
    filepath: str,
    dst_path: str,
    seed: int,
    teacher: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    length: int = 4096,
    card_id: int = 0,
    total_cards: int = 9,
    num_for_card: int = 80000,
    batch_size: int = 4,
    write_batch: int = 800
):
    # 加载流式数据集
    dataset = load_dataset(filepath, split="train", streaming=True)
    dataset = dataset.shard(total_cards, card_id)
    dataset = dataset.shuffle(seed=seed)
    
    # 限制样本数量并包装为可计算长度的数据集
    dataset = TakeIterableDataset(dataset, num_for_card)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 批量处理函数
    def batched_compute_teacher_logits(examples):
        # 批量tokenize
        tokenized = tokenizer(
            [ex['text'] for ex in examples],
            max_length=length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].to(teacher.device)
        attention_mask = tokenized['attention_mask'].to(teacher.device)
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            logits = teacher(input_ids).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=10, dim=-1)
        
        # 转换为numpy并拆分batch
        results = []
        for i in range(input_ids.size(0)):
            results.append({
                'input_ids': input_ids[i].cpu().tolist(),
                'attention_mask': attention_mask[i].cpu().tolist(),
                'topk_prob': topk.values[i].cpu().tolist(),
                'topk_idx': topk.indices[i].cpu().tolist()
            })
        return results

    # 创建保存目录
    os.makedirs(dst_path, exist_ok=True)
    file_num = 0
    buffer = []

    # 使用DataLoader处理数据集
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        drop_last=False
    )

    # 逐批次处理
    for examples in tqdm(dataloader, desc=f"Processing card {card_id}", total=len(dataset)//batch_size + 1):
        # 计算当前已处理样本数
        processed_samples = file_num * write_batch + len(buffer)
        if processed_samples >= num_for_card:
            break
        
        # 处理当前批次
        processed_batch = batched_compute_teacher_logits(examples)
        buffer.extend(processed_batch)
        
        # 当缓冲区达到写入batch时保存
        while len(buffer) >= write_batch:
            # 计算实际需要写入的数量
            write_size = min(write_batch, num_for_card - file_num * write_batch)
            
            # 截断buffer到需要写入的数量
            write_data = buffer[:write_size]
            
            # 保存数据
            df = pd.DataFrame(write_data)
            df.to_parquet(os.path.join(dst_path, f"part_{card_id}_{file_num}.parquet"))
            
            # 更新buffer和计数器
            buffer = buffer[write_size:]
            file_num += 1
            
            # 检查是否完成所有样本处理
            if file_num * write_batch >= num_for_card:
                break

    # 处理剩余数据
    remaining_samples = num_for_card - file_num * write_batch
    if remaining_samples > 0 and buffer:
        df = pd.DataFrame(buffer[:remaining_samples])
        df.to_parquet(os.path.join(dst_path, f"part_{card_id}_{file_num}.parquet"),
                    compression='zstd',
                    engine='pyarrow')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--card_id", type=int, default=0)
    parser.add_argument("--total_cards", type=int, default=8)
    parser.add_argument("--num_for_card", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--write_batch", type=int, default=800)
    args = parser.parse_args()

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(teacher_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).eval().to("cuda")

    prepare_slimpajamar_dataset_for_distillation(
        filepath=data_path,
        dst_path=dst_path,
        seed=42,
        teacher=model,
        tokenizer=tokenizer,
        card_id=args.card_id,
        total_cards=args.total_cards,
        num_for_card=args.num_for_card,
        batch_size=args.batch_size,
        write_batch=args.write_batch
    )

'''
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 0 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 1 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=2 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 2 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 3 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 4 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=5 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 5 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 6 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
unset LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=7 /opt/conda/envs/llmf/bin/python /chenyaofo/chenyf/LLaMA-Factory/script-for-distillation/create.py --card_id 7 --total_cards 8 --num_for_card 96000 --batch_size 2 --write_batch 800
'''