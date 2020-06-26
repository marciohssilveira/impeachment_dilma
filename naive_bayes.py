import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report

data = pd.read_csv('impeachment_br.csv').dropna()


def trata_voto(x):
    if x == 'Sim':
        return 1
    else:
        return 0


data['Voto'] = data['Voto'].apply(trata_voto)


vectorizer = CountVectorizer()

features = vectorizer.fit_transform(data['Fala'])

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    data['Voto'],
                                                    test_size=0.3,
                                                    random_state=88)

multinomial_nb = MultinomialNB().fit(X_train, y_train)
# gaussian_nb = GaussianNB().fit(X_train, y_train)
bernoulli_nb = BernoulliNB().fit(X_train, y_train)

y_pred_multinomial = multinomial_nb.predict(X_test)
# y_pred_gaussian = gaussian_nb.predict(X_test)
y_pred_bernoulli = bernoulli_nb.predict(X_test)

print(classification_report(y_test, y_pred_multinomial))
# print(classification_report(y_test, y_pred_gaussian))
print(classification_report(y_test, y_pred_bernoulli))


example = ['deus familia militares', 'democracia estado direito', 'democracia deus familia']
doc_term_matrix = vectorizer.transform(example)
y_pred_example = naive_bayes.predict_proba(doc_term_matrix)
print(y_pred_example)