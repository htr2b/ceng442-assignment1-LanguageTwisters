import pandas as pd
from pathlib import Path
import logging
from gensim.models import Word2Vec, FastText

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

def main():
    INPUT_DIR = Path("cleaned_data")
    OUTPUT_DIR = Path("embeddings")
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = [
        INPUT_DIR / "labeled-sentiment_2col.xlsx",
        INPUT_DIR / "test__1__2col.xlsx",
        INPUT_DIR / "train__3__2col.xlsx",
        INPUT_DIR / "train-00000-of-00001_2col.xlsx",
        INPUT_DIR / "merged_dataset_CSV__1__2col.xlsx",
    ]

    sentences=[]
    for f in files:
        if not f.exists(): 
            print(f"[WARN] Missing: {f}")
            continue
        df=pd.read_excel(f, usecols=["cleaned_text"])
        sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())

    if not sentences:
        print("ERROR: No sentences found.")
        return

    print(f"Training Word2Vec on {len(sentences)} samples...")
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=8,
        min_count=2,
        sg=1,
        negative=15,
        epochs=25,
        workers=4
    )
    (OUTPUT_DIR / "word2vec.model").parent.mkdir(exist_ok=True, parents=True)
    w2v.save(str(OUTPUT_DIR / "word2vec.model"))

    print("Training FastText...")
    ft = FastText(
        sentences=sentences,
        vector_size=300,
        window=8,
        min_count=2,
        sg=1,
        min_n=3,
        max_n=6,
        epochs=25,
        workers=4
    )
    ft.save(str(OUTPUT_DIR / "fasttext.model"))
    print("Models saved to 'embeddings/'.")

if __name__ == "__main__":
    main()
