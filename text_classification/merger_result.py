import pandas as pd
import numpy as np
sub=pd.read_csv('f:/subject.csv')
value=pd.read_csv('f:/value.csv')

result = pd.merge(sub, value, on=['content_id', 'content'], how='left')
result['sentiment_word'] = np.nan
result = result[['content_id', 'subject', 'sentiment_value', 'sentiment_word']]
result.to_csv('f:/result.csv',index=False)