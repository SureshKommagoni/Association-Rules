# implmenting apriori algorithm form mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

movies = []

with open("D:\ExcelR\Assignments\Association Rules\my_movies.csv") as f:
    movies = f.read()
    
# splitting the data into separate transactions using separator as "\n"

movies = movies.split("\n")
movies_list = []

for i in movies:
    movies_list.append(i.split(","))
    
all_movies_list = [i for item in movies_list for i in item]

from collections import Counter, OrderedDict

item_frequencies = Counter(all_movies_list)

# after sorting

item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

movies_series = pd.DataFrame(pd.Series(movies_list))
movies_series = movies_series.iloc[:11,:]

movies_series.columns = ["transactions"]

X = movies_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support = 0.005, max_len =3, use_colnames = True)

# Most Frequent item sets based on support

frequent_itemsets.sort_values('support', ascending = False, inplace = True)


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)









