'''Реалізація моделі "мішка слів"(bag of words) на мові Python.
В цьому прикладі використовуємо бібліотеку Beautifulsoup4 для аналізу данних з Вікіпедії,
Urlib для відкриття та обробки url-адресів, re для регулярних виразів та NLTK для обробки природньої мови'''

import nltk
import bs4 as bs
import urllib.request
import re
# отримуємо URL-адресу html файла
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Python_(programming_language)')
raw_html = raw_html.read()
# HTML-парсер в lxml
article_html = bs.BeautifulSoup(raw_html, 'lxml')
# З необробленого HTML ми фільтруємо текст в тексті абзацу, створюємо повний корпус, об'єднавши всі абзаци.
article_paragraphs = article_html.find_all('p')
article_text = ''

for para in article_paragraphs:
    article_text += para.text
# розбиваємо корпус на окремі речення. Для цього скористаємося функцією sent_tokenize з бібліотеки NLTK.
corpus = nltk.sent_tokenize(article_text)
# Перебираємо кожне речення в корпусі, перетворимо пропозицію в нижній регістр, а потім видаляємо розділові знаки і порожні прогалини з тексту.
for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])

# створюємо словник, в якому в якості ключа будуть всі слова, які були в тексті, а значення - кількість повторень в тексті
bag_of_words = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in bag_of_words.keys():
            bag_of_words[token] = 1
        else:
            bag_of_words[token] += 1

# створюємо список, в якому будуть слова та кількість повторень та сортуємо його за спаданням
pairs = []
list_of_words = bag_of_words.keys()
for i in list_of_words:
    pairs.append([i, bag_of_words[i]])

sorted_bag_of_words = reversed(sorted(pairs, key=lambda sort: sort[1]))
for i in sorted_bag_of_words:
    print(f"{i[0]} - {i[1]}", end=";\t")


