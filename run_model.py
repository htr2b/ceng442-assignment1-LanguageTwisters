import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec, FastText
from numpy import dot
from numpy.linalg import norm
import math

def cos(a,b): 
    return float(dot(a,b)/(norm(a)*norm(b))) if (norm(a)>0 and norm(b)>0) else float("nan")

def pair_sim(model, pairs):
    vals=[]
    for a,b in pairs:
        try:
            vals.append(model.wv.similarity(a,b))
        except KeyError:
            pass
    return sum(vals)/len(vals) if vals else float("nan")

def lexical_coverage(model, tokens):
    vocab=model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab)/max(1,len(tokens))

def read_tokens(path):
    try:
        df=pd.read_excel(path, usecols=["cleaned_text"])
        return [t for row in df["cleaned_text"].astype(str) for t in row.split()]
    except Exception as e:
        print(f"[SKIP] {path}: {e}")
        return []

def neighbors(model, word, k=5):
    try: return [w for w,_ in model.wv.most_similar(word, topn=k)]
    except KeyError: return []

def main():
    w2v=Word2Vec.load("embeddings/word2vec.model")
    ft =FastText.load("embeddings/fasttext.model")
    files=[
        Path("cleaned_data/labeled-sentiment_2col.xlsx"),
        Path("cleaned_data/test__1__2col.xlsx"),
        Path("cleaned_data/train__3__2col.xlsx"),
        Path("cleaned_data/train-00000-of-00001_2col.xlsx"),
        Path("cleaned_data/merged_dataset_CSV__1__2col.xlsx"),
    ]

    print("== Lexical Coverage ==")
    for f in files:
        toks=read_tokens(f)
        if not toks: 
            continue
        print(f"{f.name}: W2V={lexical_coverage(w2v,toks):.3f}, FT={lexical_coverage(ft,toks):.3f}")

    syn_pairs=[("yaxşı","əla"),("bahalı","qiymətli"),("ucuz","sərfəli"),("gözəl","qəşəng")]
    ant_pairs=[("yaxşı","pis"),("bahalı","ucuz"),("gözəl","çirkin")]

    syn_w2v=pair_sim(w2v,syn_pairs); syn_ft=pair_sim(ft,syn_pairs)
    ant_w2v=pair_sim(w2v,ant_pairs); ant_ft=pair_sim(ft,ant_pairs)

    print("\n== Similarity ==")
    print(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
    print(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
    sep_w2v = syn_w2v - ant_w2v if (not math.isnan(syn_w2v) and not math.isnan(ant_w2v)) else float('nan')
    sep_ft  = syn_ft  - ant_ft  if (not math.isnan(syn_ft ) and not math.isnan(ant_ft )) else float('nan')
    print(f"Separation: W2V={sep_w2v:.3f}, FT={sep_ft:.3f}")

    seed_words=["yaxşı","pis","çox","bahalı","ucuz","mükəmməl","dəhşət","<PRICE>","<RATING_POS>"]
    print("\n== Nearest Neighbors ==")
    for w in seed_words:
        print(f"W2V NN for '{w}': {neighbors(w2v,w)}")
        print(f"FT  NN for '{w}': {neighbors(ft ,w)}")

if __name__ == "__main__":
    main()
