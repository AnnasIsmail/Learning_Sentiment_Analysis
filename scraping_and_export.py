import numpy as np
import pandas as pd
from google_play_scraper import Sort, reviews

result, continuation_token = reviews(
    'com.openai.chatgpt',
    lang='id',  # defaults to 'en'
    country='id',  # defaults to 'us'
    # defaults to Sort.MOST_RELEVANT you can use Sort.NEWEST to get newst reviews
    sort=Sort.NEWEST,
    count=100000,  # defaults to 100
    # defaults to None(means all score) Use 1 or 2 or 3 or 4 or 5 to select certain score
    filter_score_with=None
)
df_result = pd.DataFrame(np.array(result), columns=['review'])
df_result = df_result.join(pd.DataFrame(df_result.pop('review').tolist()))
df_result = df_result[['userName', 'score', 'at', 'content']]


def labeling(score):
    if score < 3:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    elif score > 3:
        return 'Positive'


df_result['Label'] = df_result['score'].apply(labeling)
df_result.dropna(subset=['Label'], inplace=True)

df_result.to_csv('data_export.csv', index=False)
df_result.to_excel('data_export.xlsx', index=False)

tanggal_paling_lama = df_result['at'].min()
print("Tanggal Paling Lama:", tanggal_paling_lama)

print(df_result.shape[0])
print(df_result.head())
