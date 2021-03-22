import pandas

path = '/data20.04/data/MC_OCR/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv'
with open(path, 'r', encoding='utf-8') as f:
    tex = f.readlines()

print(tex[1])
csvfile = pandas.read_csv(path, encoding='utf-8', header=0)
print(csvfile.columns.values)
a = csvfile[:]['anno_texts'].values.tolist()
print(a)
