# %%
from airtable import Airtable
import pandas as pd

base_id = 'app5x8nFdXg0Jj9JY'
table_name = '评分'
api_key = 'keyFvA03TJKPauBhh'

AT = Airtable(base_id, table_name, api_key)
print(AT)

# %%
view_name = 'Grid view'
records = AT.get_all(view=view_name)

df = pd.DataFrame.from_records((r['fields'] for r in records))
photos = df['房源照片']
ranks = df['平均评分']

# %%
import requests

def get_image(url, pic_name):
    response = requests.get(url)
    with open(pic_name, "wb") as fp:
        for data in response.iter_content(128):
            fp.write(data)

# %%
s = len(photos)
for i in range(s):
    pic_name = './data/%04d.png'%i
    url = photos[i][0]['url']
    get_image(url, pic_name)

# %%
with open('./data/label.txt', "w") as fopen:
    for i in range(s):
        pic_name = './data/%04d.png'%i
        fopen.write(pic_name)
        fopen.write('\t')
        fopen.write(str(ranks[i]))
        fopen.write('\n')


# %%
