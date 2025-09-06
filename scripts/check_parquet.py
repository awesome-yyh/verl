import pyarrow.parquet as pq

class ParquetDataset():
    def __init__(self, parquet_path):
        self.table = pq.read_table(parquet_path)
        self.length = self.table.num_rows

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.table.slice(index, 1).to_pydict()
        # 根据需要转换数据格式
        return {k: v for k, v in row.items()}


if __name__ == "__main__":
    parquet_path = "/data/app/yangyahe/base_datasets/gsm8k/train.parquet"
    parquet_path = "/data/app/yangyahe/base_datasets/gsm8k/test.parquet"
    parquet_path = "data/anli_ft.parquet"
    parquet_path = "data/anli_ft0.5.parquet"
    parquet_path = "data/anli_ft_test.parquet"
    parquet_path = "data/anli_ft_test_system.parquet"
    parquet_path = "data/anli_ft_train_system_tgt_add_src.parquet"
    # parquet_path = "data/anli_ft_test_system_tgt_add_src.parquet"
    dataset = ParquetDataset(parquet_path)
    
    # 单条数据访问
    print("\nSample item at index 42:")
    print(dataset[42])
    
    # 批量迭代
    print("\nIterating first 5 items:")
    for i, item in enumerate(dataset):
        if i >= 5:
            break
        print(item)
    
    print(f"len: {len(dataset)}")
    