# implimenting apriori algorithm from mlxtend

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

book = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Association Rules/book.csv")

book.head(10)

book.isnull().sum()

book.describe() 

frequent_itemsets = apriori(book, min_support=0.005, max_len =3, use_colnames = True)

# most frequent itemsets based on support

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

















    
    
    
    
    
    
    
    
    
    
    
    
    

   
    
    
