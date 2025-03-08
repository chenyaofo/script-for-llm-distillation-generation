from datasets import load_dataset, DatasetDict
import os
import shutil

def merge_parquet_shards(
    src_dir,
    output_dir,
    num_shards=32,
    shard_compress=True
):
    # 1. 收集所有原始分片文件
    parquet_files = [
        os.path.join(src_dir, f) 
        for f in os.listdir(src_dir) 
        if f.startswith('part_') and f.endswith('.parquet')
    ]

    # parquet_files = parquet_files[:8]
    
    total_size = sum(os.path.getsize(f) for f in parquet_files)
    
    # 转换为人类可读的格式
    size = total_size
    unit = 'B'
    for u in ['KB', 'MB', 'GB', 'TB']:
        if size >= 1024:
            size /= 1024
            unit = u
        else:
            break
    
    print(f"Total size of {len(parquet_files)} files: {size:.2f} {unit}")

    # 2. 加载完整数据集
    dataset = load_dataset(
        'parquet',
        data_files={'train': parquet_files},
        split='train',
        num_proc=8
    )
    
    # 3. 强制清理输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # 4. 手动分片并保存（控制文件名格式）
    for shard_idx in range(num_shards):
        # 分割数据集
        shard = dataset.shard(
            num_shards=num_shards,
            index=shard_idx,
            contiguous=True  # 确保分片连续划分
        )
        
        # 生成文件名（格式：train-00000-of-00032.parquet）
        filename = f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        file_path = os.path.join(output_dir, filename)
        
        # 保存分片（支持压缩）
        shard.to_parquet(
            file_path,
            compression="zstd" if shard_compress else None
        )


if __name__ == "__main__":
    merge_parquet_shards(
        src_dir='/chenyaofo/datasets/slimpajama-dsv2-distill/parquet_files',
        output_dir='/chenyaofo/datasets/slimpajama-dsv2-distill/unified_dataset2',
        num_shards=500,
        shard_compress=True
    )