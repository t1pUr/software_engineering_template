'''Розбір моделі word2vec на основі датасету російськомовних Twitter-постів.
Бібліотеки, які знадобляться: NLTK для обробки природньої мови, re для регулярних виразів, Pandas для аналізу даних(в нашому випадку зчитування .csv файлів,
pymorphy2 для приведення слова до нормальної форми та Gensim для роботи з векторними моделями (в нашому випадку word2vec).
'''

import nltk
import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
#Отримаємо стоп-слова від NLTK та зчитаємо файли за допомогою Pandas
nltk.download('stopwords')

df_pos = pd.read_csv("positive.csv", sep=";", header=None)
df_neg = pd.read_csv("negative.csv", sep=";", header=None)
# з'єднаємо позитивні та негативні твіти
df = df_pos.iloc[:, 3].append(df_neg.iloc[:, 3])
df = df.dropna().drop_duplicates()
#нормалізуємо дані (позбуваємося знаків пунктуації, букв латинського алфавіту, розбиваємо пост на токени, лематизація, видалення стоп-слова)
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None
# отримуємо датасет за допомогою метода apply() 
data = df.apply(lemmatize)
data = data.dropna()
# ініціалізуємо модель Word2vec та отримаємо словник
w2v_model = Word2Vec(
    min_count=10,
    window=2,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.0007,
    sample=6e-5,
    sg=1)

w2v_model.build_vocab(data)
# навчаємо модель
w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

# приклади, за допомогою яких можна знаходити схожі  слова
print(w2v_model.wv.most_similar(positive=["любить"]))
print(w2v_model.wv.most_similar(positive=["мужчина"]))
print(w2v_model.wv.most_similar(positive=["день", "завтра"]))
print(w2v_model.wv.most_similar(positive=["папа", "брат"], negative=["мама"]))
