from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

items = []

total_tokens = defaultdict(int)

for path_to_file in tqdm(os.listdir("abc")):
    file_tokens = defaultdict(int)

    # open the file
    with open(f"abc/{path_to_file}", "r") as file:
        tokens = file.read().split(" ")

        # split file_tokens by space and remove empty strings
        tokens = list(filter(None, tokens))

        # count the number of each token
        for token in tokens:
            file_tokens[token] += 1
            total_tokens[token] += 1

    # finally, append a tuple with the filename and the file_tokens
    items.append((path_to_file, file_tokens))

for tok in sorted(total_tokens.items(), key=lambda x: x[1], reverse=True):
    print(tok)
print(len(total_tokens))

"""
# find the total number of unique tokens
unique_tokens = set()
for item in items:
    unique_tokens.update(item[1].keys())

# add "filename" as a column
unique_tokens.add("filename")

# make a dataframe with the unique tokens as columns
df = pd.DataFrame(columns=list(unique_tokens))

# add the counts of each token to the dataframe
dataframes = []  # create a list to store all the dataframes

for item in items:
    # create dataframe with index as filename
    temp_df = pd.DataFrame(item[1], index=[item[0]])
    dataframes.append(temp_df)

# concatenate all dataframes
df = pd.concat(dataframes, ignore_index=False)

# fill all NaN values with 0
df = df.fillna(0)

# sum over axis=0 and sort by frequency
df.sum(axis=0).sort_values(ascending=False).plot(kind="bar", figsize=(15, 4))
# set the x-axis font size to 12
plt.xticks(fontsize=6)
# set the x axis label to 'token'
plt.xlabel("Token", fontsize=18)
# set the y axis label to 'count'
plt.ylabel("Count", fontsize=18)
# set y-axis to log scale
plt.yscale("log")
# rotate the x axis labels 45 degrees
plt.xticks(rotation=45)
# remove the legend
plt.legend().remove()
# plt.show()
"""
