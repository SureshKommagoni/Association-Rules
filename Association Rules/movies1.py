# implementing apriori algorithm from mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

movies = pd.read_csv("file:///D:/ExcelR/Assignments/Association Rules/my_movies.csv")
movies_new = movies.iloc[:,[5,6,7,8,9,10,11,12,13,14]]

frequent_itemsets = apriori(movies_new, min_support=0.005, max_len =3, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)



movies_new = movies.iloc[,6:15]
movies.dropna()
movies.isnull().sum()

movies = movies[,6:15]
# with open("D:\\ExcelR\\Assignments\\Association Rules\\my_movies.csv") as f:
  #  movies = f.read()








