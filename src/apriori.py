import pandas as pd
import numpy as np
import datetime
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

with open("top_n-20m.pickle", "rb") as fp:
    top_n = pickle.load(fp)
top_n_items = [ [x[0] for x in row] for row in top_n]

te = TransactionEncoder()
te_ary = te.fit(top_n_items).transform(top_n_items, sparse=True)
topn_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
print("Sparse df created")

apriori_start_time = datetime.datetime.now()
frequent_itemsets = apriori(topn_df, min_support=0.002, verbose=1, low_memory=True, use_colnames=True)
apriori_end_time = datetime.datetime.now()
print("Training duration: " + str(apriori_end_time - apriori_start_time))

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
rules = association_rules(frequent_itemsets)

with open("association-rules-20m.pickle", "wb+") as fp:
    pickle.dump(rules, fp)