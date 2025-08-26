import pandas as pd


# merge_path="data/anli_ft_correct_merge_train_shuffled_shuffled.parquet"
merge_path="data/anli_ft_correct_merge_test_shuffled_shuffled.parquet"

df2 = pd.read_parquet(merge_path)
print(df2.shape)
print(df2.head)