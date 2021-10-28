from amethyst.dataloader import dataset, split
import pandas as pd


df = pd.read_csv("./data/movielens100k.csv")

train, test = split.stratified_split(df, 0.8, "userID", "itemID", 
                                    filter_col="item")


train_data = dataset.Dataloader.dataloader(train.itertuples(index=False))
test_data = dataset.Dataloader.dataloader(test.itertuples(index=False))

print(f"Number of users: {train_data.user_count}")
print(f"Number of users: {test_data.user_count}")


