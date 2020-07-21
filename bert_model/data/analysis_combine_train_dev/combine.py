import pandas as pd

# f1 = pd.read_csv('task5_training.tsv', sep='\t', encoding='ISO-8859-1')
# f2 = pd.read_csv('task5_validation.tsv', sep='\t', encoding='ISO-8859-1')
# file = [f1,f2]
# train = pd.concat(file)
# train.to_csv("newtrain" + ".csv", index=0)


# train_df=pd.read_csv("newtrain.csv")
# train_df['class'] = (train_df['class']-1)
# train_df.to_csv("newnewtrain" + ".csv", index=0)

f1 = pd.read_csv('task5_test_participant.tsv', sep='\t')
f1.to_csv("test" + ".csv", index=0)
