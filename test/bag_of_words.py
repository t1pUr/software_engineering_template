import nltk
import bs4 as bs
import urllib.request
import re

raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Python_(programming_language)')
raw_html = raw_html.read()
article_html = bs.BeautifulSoup(raw_html, 'lxml')
article_paragraphs = article_html.find_all('p')
article_text = ''

for para in article_paragraphs:
    article_text += para.text

corpus = nltk.sent_tokenize(article_text)

for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])

bag_of_words = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in bag_of_words.keys():
            bag_of_words[token] = 1
        else:
            bag_of_words[token] += 1

pairs = []
list_of_words = bag_of_words.keys()
for i in list_of_words:
    pairs.append([i, bag_of_words[i]])

sorted_bag_of_words = reversed(sorted(pairs, key=lambda sort: sort[1]))
for i in sorted_bag_of_words:
    print(f"{i[0]} - {i[1]}", end=";\t")
