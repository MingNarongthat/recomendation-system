import pandas as pd 
import numpy as np
from pandas.core.frame import DataFrame 
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data_clean.csv')

user_pivot = data.pivot(index='firm',columns='name',values='Rating')
item_pivot = user_pivot.T
item_pivot.fillna(0, inplace=True)
similarities = cosine_similarity(item_pivot)
cosine_similarity_df = pd.DataFrame(similarities,index=item_pivot.index,columns=item_pivot.index)
print(cosine_similarity_df.columns)
cosine_similarity_series = cosine_similarity_df.loc['Pop socket']
ordered_similarities = cosine_similarity_series.sort_values(ascending=False)
print(ordered_similarities)





