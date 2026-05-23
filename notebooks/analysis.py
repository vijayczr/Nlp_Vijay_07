#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os


sys.path.append(os.path.abspath(".."))


# In[2]:


from src.journal_api import get_papers_dataframe


# In[3]:


df = get_papers_dataframe(
    venue="Transactions of the Association for Computational Linguistics",
    fields_of_study="Computer Science",
    publication_types="JournalArticle",
    date_range="2010-01-01:2025-12-31"
)

df.head()


# In[4]:


from src.journal_api import get_papers_dataframe


# In[5]:


from src.embeddings import EmbeddingModel

embedder = EmbeddingModel()


# In[6]:


from src.scoring import compute_similarity, summary_statistics


# In[7]:


journal_scope = """
This journal publishes research in computational linguistics,
natural language processing, machine learning for language understanding,
syntax, semantics, discourse, multilingual NLP, and language generation.
"""


# In[8]:


scope_embedding = embedder.encode_text(journal_scope)

article_embeddings = embedder.encode_list(
    df["abstract"].fillna("").tolist()
)


# In[9]:


scores = compute_similarity(article_embeddings, scope_embedding)

df["alignment_score"] = scores
df.head()


# In[10]:


summary_statistics(scores)


# In[11]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.hist(df["alignment_score"], bins=20)
plt.title("Distribution of Alignment Scores")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Papers")
plt.show()


# In[12]:


yearly_scores = df.groupby("year")["alignment_score"].agg(
    mean="mean",
    median="median",
    count="count",
    std="std"
).reset_index()

yearly_scores


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(yearly_scores["year"], yearly_scores["mean"], marker="o")
plt.title("Average Thematic Alignment by Year")
plt.xlabel("Year")
plt.ylabel("Mean Alignment Score")
plt.grid(True, alpha=0.3)
plt.show()


# In[14]:


low = df.sort_values("alignment_score").head(10)
low[["year","title","alignment_score"]]


# In[15]:


high = df.sort_values("alignment_score", ascending=False).head(10)
high[["year","title","alignment_score"]]


# In[16]:


papers_per_year = df["year"].value_counts().sort_index()

plt.figure(figsize=(12,5))
papers_per_year.plot(kind="bar")
plt.title("Number of Papers Published per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


# In[17]:


plt.figure(figsize=(12,6))
plt.plot(yearly_scores["year"], yearly_scores["mean"], marker="o", linewidth=2)
plt.fill_between(
    yearly_scores["year"],
    yearly_scores["mean"] - yearly_scores["std"].fillna(0),
    yearly_scores["mean"] + yearly_scores["std"].fillna(0),
    alpha=0.2
)
plt.title("Average Alignment by Year (± Std Dev)")
plt.xlabel("Year")
plt.ylabel("Mean Alignment")
plt.grid(alpha=0.3)
plt.show()


# In[19]:


df.sort_values("alignment_score", ascending=False)[["year","title","alignment_score"]].head(10)


# In[20]:


df.sort_values("alignment_score")[["year","title","alignment_score"]].head(10)


# In[21]:


yearly_scores["count"].corr(yearly_scores["mean"])


# In[22]:


## Representation


# In[23]:


papers_per_year = df["year"].value_counts().sort_index()
papers_per_year


# In[24]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
papers_per_year.plot(kind="bar")
plt.title("Number of Papers Published Per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[25]:


plt.figure(figsize=(10,5))
plt.hist(df["alignment_score"], bins=25, edgecolor="black")
plt.axvline(df["alignment_score"].mean(), linestyle="--", linewidth=2, label="Mean")
plt.axvline(df["alignment_score"].median(), linestyle=":", linewidth=2, label="Median")
plt.title("Distribution of Alignment Scores")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Papers")
plt.legend()
plt.show()


# In[26]:


low_papers = df.sort_values("alignment_score").head(10)
low_papers[["year", "title", "alignment_score"]]


# In[27]:


high_papers = df.sort_values("alignment_score", ascending=False).head(10)
high_papers[["year", "title", "alignment_score"]]


# In[28]:


plt.figure(figsize=(14,6))
df.boxplot(column="alignment_score", by="year", grid=False, rot=45)
plt.title("Alignment Score Distribution by Year")
plt.suptitle("")
plt.xlabel("Year")
plt.ylabel("Alignment Score")
plt.show()


# In[29]:


merged = yearly_scores.merge(
    papers_per_year.rename("count"),
    left_on="year",
    right_index=True
)


# In[30]:


print(merged.columns)


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

sns.heatmap(
    merged[["mean", "median", "count_x", "std"]].corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Between Yearly Metrics")
plt.show()


# In[32]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)

pca_result = pca.fit_transform(article_embeddings)

df["pca1"] = pca_result[:, 0]
df["pca2"] = pca_result[:, 1]


# In[33]:


plt.figure(figsize=(12,8))

plt.scatter(
    df["pca1"],
    df["pca2"],
    alpha=0.7
)

plt.title("Semantic Map of Journal Articles (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(alpha=0.3)
plt.show()


# In[34]:


plt.figure(figsize=(12,8))

scatter = plt.scatter(
    df["pca1"],
    df["pca2"],
    c=df["alignment_score"],
    alpha=0.8
)

plt.colorbar(scatter, label="Alignment Score")

plt.title("Semantic Distribution Colored by Alignment")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid(alpha=0.3)
plt.show()


# In[35]:


# Methodology

## Data Source
## The dataset was collected using the Semantic Scholar API.  
## The selected journal was *Transactions of the Association for Computational Linguistics (TACL)*.

## Textual Inputs
## For each publication, the article abstract was used as the textual representation.  
## The journal's aims and scope statement was used as the thematic reference document.

## NLP Representation
## Texts were converted into dense semantic embeddings using the Sentence-BERT model:

## `all-MiniLM-L6-v2`

## Similarity Measure
## Cosine similarity was computed between each article embedding and the journal scope embedding.

## Objective
## The objective is to evaluate how strongly published articles align with the stated thematic focus of the journal.


# In[36]:


## PHASE 2


# In[37]:


get_ipython().system('pip install keybert')


# In[38]:


from keybert import KeyBERT

kw_model = KeyBERT()

keywords = kw_model.extract_keywords(
    journal_scope,
    keyphrase_ngram_range=(1,2),
    stop_words="english",
    top_n=8
)

keywords


# In[39]:


labels = [k[0] for k in keywords]
labels


# In[40]:


from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# In[41]:


sample_text = df["abstract"].dropna().iloc[0]

classifier(sample_text, labels)


# In[42]:


labels


# In[43]:


clean_labels = [
    "Computational Linguistics",
    "Natural Language Processing",
    "Language Processing",
    "Multilingual NLP",
    "Language Generation",
    "Linguistics"
]


# In[ ]:




