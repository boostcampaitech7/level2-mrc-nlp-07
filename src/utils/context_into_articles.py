import os
import datasets
from datasets import load_dataset
import json

# corpus 파일 읽어오기
corpus_path = './기본 데이터/wikipedia_documents.json'

with open(corpus_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

wiki_listed = list(wiki.items())

wiki_listed_test = wiki_listed[:10]

titles = []
texts = []

count = 0
text_agg = ""

for index in range(len(wiki_listed_test)):
    d = wiki_listed_test[index][1]
    title = d['title']
    text = d['text']

    if not titles:
        titles.append(title)
        texts.append(text)

temp_dict = dict()

for i in wiki_listed:
    d = i[1]
    if d["title"] in temp_dict.keys():
        text_A = temp_dict[d["title"]]['text']
        text_B = d['text']
        text_C = text_A + text_B
        temp_dict[d["title"]]['text'] = text_C
    else:
        temp_dict[d["title"]] = d

wiki_articles = []
count = 0
for i in temp_dict.values():
    wiki_articles.append([count, i])
    count += 1

wiki_articles = dict(wiki_articles)

for k, v in tuple(wiki_articles.items()):
    print(k, v)    

with open('wikipedia_documents_articles.json', 'w', encoding = "utf-8") as json_file:
    json.dump(wiki_articles, json_file, ensure_ascii=False, indent=4) 