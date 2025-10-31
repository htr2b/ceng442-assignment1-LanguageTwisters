import re, html, unicodedata, pandas as pd
from pathlib import Path
from ftfy import fix_text

def lower_az(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I", "ı").replace("İ", "i")
    return s.lower().replace("i̇", "i")

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
USER_RE = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1+")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS = re.compile(r"(.)\1{2,}", re.UNICODE)
TOKEN_RE = re.compile(r"[A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq]+|<NUM>|URL|EMAIL|PHONE|USER|EMO_(?:POS|NEG)")

EMO_MAP = {
    "🙂":"EMO_POS","😀":"EMO_POS","😍":"EMO_POS","😊":"EMO_POS","👍":"EMO_POS",
    "☹":"EMO_NEG","🙁":"EMO_NEG","😠":"EMO_NEG","😡":"EMO_NEG","👎":"EMO_NEG"
}
SLANG_MAP = {"slm":"salam","tmm":"tamam","sagol":"sağol","cox":"çox","yaxsi":"yaxşı"}
NEGATORS = {"yox","deyil","heç","qətiyyən","yoxdur"}

NEWS_HINTS = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂)")
REV_HINTS = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

PRICE_RE = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE = re.compile(r"\bçox yaxşı\b")
NEG_RATE = re.compile(r"\bçox pis\b")

def detect_domain(text: str) -> str:
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s): return "reviews"
    return "general"

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    if domain == "reviews":
        s = PRICE_RE.sub(" <PRICE> ", cleaned)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", cleaned)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        return " ".join(s.split())
    return cleaned

def normalize_text_az(s: str, numbers_to_token=True, keep_sentence_punct=False) -> str:
    if not isinstance(s, str): return ""
    for emo, tag in EMO_MAP.items(): s = s.replace(emo, f" {tag} ")
    s = fix_text(html.unescape(s))
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = re.sub(r"#([A-Za-z0-9_]+)", lambda m: " " + re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1)) + " ", s)
    s = USER_RE.sub(" USER ", s)
    s = lower_az(s)
    s = MULTI_PUNCT.sub(r"\1", s)
    if numbers_to_token: s = re.sub(r"\d+", " <NUM> ", s)
    s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]" if keep_sentence_punct else r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", " ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    toks = TOKEN_RE.findall(s)
    norm, mark_neg = [], 0
    for t in toks:
        t = SLANG_MAP.get(REPEAT_CHARS.sub(r"\1\1", t), t)
        if t in NEGATORS:
            norm.append(t); mark_neg = 3; continue
        if mark_neg > 0 and t not in {"URL","EMAIL","PHONE","USER"}:
            norm.append(t + "_NEG"); mark_neg -= 1
        else:
            norm.append(t)
    norm = [t for t in norm if not (len(t)==1 and t not in {"o","e"})]
    return " ".join(norm).strip()

def map_sentiment_value(v, scheme: str):
    if scheme == "binary":
        try: return 1.0 if int(v)==1 else 0.0
        except: return None
    s = str(v).strip().lower()
    if s in {"pos","positive","1","müsbət","good","pozitiv"}: return 1.0
    if s in {"neu","neutral","2","neytral"}: return 0.5
    if s in {"neg","negative","0","mənfi","bad","neqativ"}: return 0.0
    return None

def process_file(in_path, text_col, label_col, scheme, out_two_col_path):
    df = pd.read_excel(in_path)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len()>0].drop_duplicates(subset=[text_col])
    df["domain"] = df[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df[text_col].astype(str).apply(normalize_text_az)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["domain"]), axis=1)
    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    out_df = df[["cleaned_text","sentiment_value","domain"]].reset_index(drop=True)
    Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False)
    dom_stats = df["domain"].value_counts().to_dict()
    print(f"Saved: {out_two_col_path} (rows={len(out_df)}, domains={dom_stats})")

def build_corpus_txt(cfg, out_txt="corpus_all.txt", domain_filter=None):
    lines=[]
    for f, text_col in cfg:
        df=pd.read_excel(f)
        for raw in df[text_col].dropna().astype(str):
            dom=detect_domain(raw)
            if domain_filter and dom!=domain_filter: continue
            s=normalize_text_az(raw,keep_sentence_punct=True)
            for p in re.split(r"[.!?]+", s):
                p=p.strip()
                if not p: continue
                p=re.sub(r"[^\w\səğıöşüçƏĞIİÖŞÜÇxqXQ]"," ",p)
                lines.append(f"dom{dom} "+p.lower())
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt,"w",encoding="utf-8") as w:
        for ln in lines: w.write(ln+"\n")
    tag = f"filtered:{domain_filter}" if domain_filter else "all domains"
    print(f"Wrote {out_txt} with {len(lines)} lines ({tag})")

def main():
    CFG=[
        ("datasets/labeled-sentiment.xlsx","text","sentiment","tri"),
        ("datasets/test__1_.xlsx","text","label","binary"),
        ("datasets/train__3_.xlsx","text","label","binary"),
        ("datasets/train-00000-of-00001.xlsx","text","labels","tri"),
        ("datasets/merged_dataset_CSV__1_.xlsx","text","labels","binary"),
    ]
    for fname,tcol,lcol,scheme in CFG:
        out=f"cleaned_data/{Path(fname).stem}_2col.xlsx"
        process_file(fname,tcol,lcol,scheme,out)
    build_corpus_txt([(c[0],c[1]) for c in CFG], "corpus_all.txt")
    build_corpus_txt([(c[0],c[1]) for c in CFG], "corpus_reviews.txt", domain_filter="reviews")

if __name__ == "__main__":
    main()
