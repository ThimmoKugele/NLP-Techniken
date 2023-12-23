#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Anwendung von NLP-Techniken

# Import der erforderlichen Bibliotheken
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

# Laden der Daten (csv.-Datei). Möglichst die Datei in selben Ordner wie den Programmcode legen
# Ansonsten Pfad evtl. entsprechend anpassen
df = pd.read_csv("rows.csv", low_memory=False)
df.shape

# Es erscheint eine Warnung welche hier ignoriert werden kann


# In[ ]:


# Der Datensatz besteht aus über eine Millionen Zeilen und 18 Spalten
# Anzeige der ersten 3 Zeilen der Daten und Transponieren des Ergebnisses
df.head(3).T
#print(df)


# In[ ]:


# Entfernung irrelevanter Daten wie Adressen oder Datum, um den Datensatz zu verkleinern
# Der kleinere Datensatz wird neu abgespeichert
df1 = df[['Product', 'Consumer complaint narrative']].copy()

# Entfernen der Datensätze, welche keine Beschwerde enthalten, also unter "Consumer complaint narrative" ein "NaN"
df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]

# Einfachere Benennung der zwei Spalten einfügen
df1.columns = ['Product', 'Complaint'] 

# Ausgabe er Anzahl übriger Beschwerden zur weiteren Verarbeitung
df1.shape

# Optional Visualisierung der Daten
# print(df)


# In[ ]:


# Der verkleinerte Datensatz besteht aus über 383000 Beschwerden und 2 Spalten 
# Optional zur Visualisierung der Daten
df1.head(5)


# In[ ]:


# Anzeige der Unterschiedlichen Kategorien (Product) aus dem Datensatz (optional)
# pd.DataFrame(df.Product.unique()).values


# In[ ]:


# Anzeige der Anzahl der Kategorien (Product) aus dem Datensatz
count_unique_products = df.Product.nunique()
print(count_unique_products)


# In[ ]:


# Hier die Anzeige der verschiedenen Kategorien und Anzahl der Beschwerden dazu in absteigender Reihenfolge
df1['Product'].value_counts()


# In[ ]:


# Da der Datensatz noch sehr groß ist und mein System an seine Grenzen in Bezug auf die Rechenleistung kommt 
# wird nun lediglich die Kategorie "Student loan" weiter verarbeitet
# Beispielhafte Darstellung der ersten 5 Einträge
df2 = df1[df1['Product'] == 'Student loan']
df2.head(5)


# In[ ]:


# Importieren der erforderlichen Bibliothek für das Stemming
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# Erstellen des Stemmers
stemmer = SnowballStemmer("english")

# Definition einer Funktion, um das Stemming anzuwenden
def stem_sentence(sentence):
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Funktion auf die Spalte 'Complaint' anwenden
df2 = df2.copy()
df2['Complaint'] = df2['Complaint'].astype(str)
df2['Complaint'] = df2['Complaint'].apply(stem_sentence)

# Visualisierung des Ergebnisses
df2.head(5)


# In[ ]:


# Als Alternative kann eine Stichprobe zur weiteren Verkleinerung des Datensazues gezogen werden
# Falls genügend Rechenkapazität zur verfügung steht kann dieser Schritt übersprungen werden
# und der Code muss entsprechend auf den Ursprungsdatensatz angepasst werden
# df2 = df2.copy(10000, random_state=1).copy()


# In[ ]:


# Alternative wenn alle Kategorien genutzt werden sollen,
# Da es ähnliche Kategorien gibt, werden diese zusammengefasst und entsprechend umbenannt
# df2.replace({'Product': 
  #            {'Credit reporting, credit repair services, or other personal consumer reports': 
  #            'Credit reporting, repair, or other', 
  #            'Credit reporting': 'Credit reporting, repair, or other',
  #           'Credit card': 'Credit card or prepaid card',
  #           'Prepaid card': 'Credit card or prepaid card',
  #           'Payday loan': 'Payday loan, title loan, or personal loan',
  #           'Money transfer': 'Money transfer, virtual currency, or money service',
  #           'Virtual currency': 'Money transfer, virtual currency, or money service'}}, 
  #          inplace= True)


# In[ ]:


# Anzeige der nun vorhandenen Kategorien
# pd.DataFrame(df2.Product.unique())


# In[ ]:


# BOW-Vektorisierung
# Import der Klasse CountVectorizer für BOW
from sklearn.feature_extraction.text import CountVectorizer

# Erstellen des CountVectorizer-Objekts
# Eingrenzung auf 100 Merkmale
# Berücksichtigung von Unigramme und Bigramme (ngram_range)
# Erstellung einer Liste mit englische und aufgrund der Daten definierten Stoppwörtern

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['xx','xxx','xxxx','00']
stopwords.extend(newStopWords)

# Entfernung der Stoppwörter aus der Liste für englische Stoppwörter (stop_words)

bow = CountVectorizer(max_features=100, min_df=1000, ngram_range=(1, 2), stop_words= stopwords)

# Umwandlung der Beschwerden in einen BoW-Vektor
bow_features = bow.fit_transform(df2.Complaint).toarray()

print(df2.columns)

#labels = df2.category_id

#Ausagbe der Ergebnisse
print("Jede der %d Beschwerden wird dargestellt durch %d Merkmale" %(bow_features.shape))


# In[ ]:


# TFIDF-Vektorisierung
# Import des TFIDF Vektorizers


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Entfernung von Wörtern, welche in weniger als 1000 Dokumenten vorkommen (min_df)
# Berücksichtigung von Unigramme und Bigramme (ngram_range)
# Entfernung der Stoppwörter aus der Liste für englische Stoppwörter (stop_words) und Datensatzspetifische Stoppwörter
# Eingrenzung auf 100 Merkmale

tfidf = TfidfVectorizer(max_features=100,sublinear_tf=True, min_df=1000,
                        ngram_range=(1, 2), 
                        stop_words=stopwords)

# Jede Beschwerde wird in einen TFIDF-Vektor umgewandelt
tfidf_features = tfidf.fit_transform(df2.Complaint).toarray()

print(df2.columns)

#labels = df2.category_id

print("Jede der %d Beschwerden wird dargestellt durch %d Merkmale" %(tfidf_features.shape))


# In[ ]:


# Wir sehen nun die Anzahl der Merkmale ist identisch, weil beide Methoden das Vokabular 
# aus den gleichen Eingabedaten erstellen. Beide Methoden betrachten jedes eindeutige Wort 
# in den Daten als ein Merkmal.
# Die Unterschiede zwischen TF-IDF und BoW liegen in der Art und Weise, wie sie die Werte für diese Merkmale berechnen


# In[ ]:


# Ausgabe von 10 Merkmalen für das erste Dokument jeweils für BOW und TF-IDF
print("BoW Merkmale für das erste Dokument: ", bow_features[20][:30])
print("TF-IDF Merkmale für das erste Dokument: ", tfidf_features[20][:30])


# In[ ]:


# Nun erkennt man, dass in großen Textsammlungen viele Nullvektoren bei BOW und TF-IDF entstehen, weshalb nochmals oben
# mittels min_df auf noch weniger Merkmale reduziert werden musste, was aber immer noch zu vielen Nullvektoren führt
# Auch zu sehen ist in dem Beispiel, dass die Werte in den BoW-Vektoren höher sind da nur die Häufigkeit jedes Wortes 
# in einem Dokument berücksichtigt wird
# während die Werte in den TF-IDF-Vektoren kleiner sind, da die Wortzählungen nach ihrer Bedeutung in den Dokumenten
# gewichtet werden


# In[ ]:


# Anzeige der 100 Merkmale
print("BoW Merkmale: ", bow.get_feature_names_out())
print("TF-IDF Merkmale: ", tfidf.get_feature_names_out())


# In[ ]:


# Die Themenanalyse wird mit den TF-IDF-Vektoren durchgeführt, da diese in der Regel bessere Ergebnisse als BOW liefert
# Berechnen der durchschnittlichen TF-IDF-Werte für jedes Wort
avg_tfidf = np.mean(tfidf_features, axis=0)

# Erstellen eines DataFrames mit den Wörtern und ihren durchschnittlichen TF-IDF-Werten
df_tfidf = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'avg_tfidf': avg_tfidf})

# Sortieren des DataFrames in absteigender Reihenfolge der durchschnittlichen TF-IDF-Werte
df_tfidf = df_tfidf.sort_values('avg_tfidf', ascending=False)

# Ausgabe der Wörter und ihrer durchschnittlichen TF-IDF-Werte
print(df_tfidf)


# In[ ]:


# Extraktion der 5 häufigsten Themen jeder Kategorie mittels sklearn LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

# Erstellung des LDA-Modells
lda_model=LatentDirichletAllocation(n_components=5,learning_method=
'online',random_state=42,max_iter=1)

# Anpassung des LDA-Models an die Merkmale
lda_top=lda_model.fit_transform(tfidf_features)

# Ausgabe der Themenverteilung für jede Beschwerde in Prozent
for i,topic in enumerate(lda_top[0]):
  print("Thema ",i,": ",topic*100,"%")
feature_names = tfidf.get_feature_names_out()
N=10
for topic_idx, topic in enumerate(lda_model.components_):
  print("\n==> Thema %d:" %(topic_idx))
  print("  * Schlüsselwörter: ", " ".join([feature_names[i] for i in topic.argsort()[:-N - 1:-1]]))


# In[ ]:


# Ausgabe der beispielhaften Themenverteilung in Prozent der ersten 5 Beschwerden
for i, topic_dist in enumerate(lda_top[:5]):
  print("\n==> Beschwerde %d:" %(i))
  print("  * Themenverteilung: ", topic_dist)


# In[ ]:


# Extraktion der 5 häufigsten Themen jeder Kategorie mittels sklearn LSA

# Erstellen des LSA-Modells
lsa_model = TruncatedSVD(n_components=5, random_state=42)

# Anpassung des LSA-Modells an die Merkmale
lsa_top = lsa_model.fit_transform(tfidf_features)

# Ausgabe der Themenverteilung für jede Beschwerde
for i, topic in enumerate(lsa_top[:5]):
    print("\n==> Thema %d:" %(i))
    print("  * Themenverteilung: ", topic)


# In[ ]:


# Ausgabe der Schlüsselwörter für jedes Thema für LSA
feature_names = tfidf.get_feature_names_out()
for topic_idx, topic in enumerate(lsa_model.components_[:5]):
    print("\n==> Thema %d:" %(topic_idx))
    print("  * Schlüsselwörter: ", " ".join([feature_names[i] for i in topic.argsort()[:-N - 1:-1]]))

