import os
import datasets
from datasets import load_dataset
import json

corpus_path = './기본 데이터/wikipedia_documents.json'

with open(corpus_path, "r", encoding="utf-8") as f:
    wiki = json.load(f)

wiki_sentences = []

for i, d in wiki.items():
    text = d['text']
    sentences = text.split('.')


    for s in sentences:
        sentence_d = d.copy()
        s.strip()

        if not wiki_sentences:
            index = 0
        else:
            index = len(wiki_sentences)

        sentence_d['text'] = s

        wiki_sentences.append([index, sentence_d])


wiki_sentences = dict(wiki_sentences)

with open('wikipedia_documents_sentences.json', 'w', encoding = "utf-8") as file:
    json.dump(wiki_sentences, file, ensure_ascii = False, indent = 4)
