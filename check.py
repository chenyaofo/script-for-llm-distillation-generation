from datasets import load_dataset

# 流式加载（适合大数据集）
ds_stream = load_dataset(
    '/chenyaofo/datasets/slimpajama-dsv2-distill/unified_dataset2',
    split='train',
    streaming=True
)
for i in range(100):
    print(f"sample {i}:", next(iter(ds_stream)).keys())
