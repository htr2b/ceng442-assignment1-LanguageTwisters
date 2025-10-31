# CENG442 Assignment 1: Advanced Preprocessing and Word Embedding Comparison for Azerbaijani

This document details the first assignment, which involved constructing an advanced, domain-specific preprocessing pipeline for the Azerbaijani language. We successfully consolidated and cleaned five different sentiment-labeled datasets. The final phase included training and rigorously evaluating both the Word2Vec and FastText embedding models based on the processed text corpus.

**Team Members:**
* `Adnan Arda Simsar`
* `Bülent Durusoy`
* `Onur Kahan`
* `Salih Yıldız`

---

## 1. Data Aggregation and Normalization Strategy

The initial phase focused on merging and standardizing data sourced from five varied Azerbaijani review and social media files. The inherent challenge lay in their inconsistent labeling (binary vs. 3-class sentiment).

We unified the sentiment labels into a continuous numerical spectrum to better represent polarity: `Negative=0.0`, `Neutral=0.5`, and `Positive=1.0`. By maintaining the `0.5` (Neutral) class, the corpus retains the ability to train future models for subtle sentiment regression tasks, rather than being confined to simple binary classification.

The final, high-quality training corpus size, post-cleaning and deduplication, reached **124,051** documents.

## 2. Comprehensive Preprocessing Pipeline Summary

Our custom preprocessing architecture was designed to handle the noise typical of web-scraped Azerbaijani text.

### **Core Transformation Steps:**
* **Custom Casing:** We implemented a tailored lowercasing function, essential for accurately mapping characters unique to Turkic languages (e.g., handling 'İ' and 'I') that standard NLP tools often mismanage.
* **Noise Reduction:** A series of filters were applied to eliminate extraneous data, including URLs, embedded HTML tags, emojis, special symbols, and redundant spacing.
* **Negation Handling:** To address the critical issue of sentiment reversal, a `_NEG` marker was introduced. Any word immediately following a negative particle (e.g., *deyil*, *yox*) was tagged (e.g., *yaxşı deyil* became *yaxşı\_NEG deyil*).
* **Emphasis Control:** To prevent exaggerated spellings from fragmenting the vocabulary, we normalized extreme repetition by capping character recurrence at two (e.g., *superrr* was corrected to *superr*).

---

## 3. Word Embedding Model Configurations

The Word2Vec (W2V) and FastText (FT) models were trained using optimized parameters to capture the semantic and morphological complexity of the Azerbaijani language.

| Parameter | Word2Vec (Skip-gram) | FastText |
| :--- | :--- | :--- |
| **Algorithm** | Skip-gram | Skip-gram |
| **Vector Size** | 100 | 100 |
| **Window Size** | 5 | 5 |
| **Min Count** | 5 | 5 |
| **Subword N-grams** | N/A | Min=3, Max=6 |
| **Epochs** | 15 | 15 |

---

## 4. Evaluation Results

The models were benchmarked using both quantitative measures (coverage and similarity) and qualitative analysis (nearest neighbors).

### 4.1. Quantitative Metrics

#### 🚀 Lexical Coverage
This metric confirms the high quality of the preprocessing stage, with both models showing near-perfect token coverage across key evaluation files.

| Evaluation File | W2V Coverage | FT Coverage |
| :--- | :--- | :--- |
| labeled-sentiment\_2col.xlsx | 0.957 | 0.957 |
| test\_\_1\_\_2col.xlsx | 1.000 | 1.000 |
| train\_\_3\_\_2col.xlsx | 1.000 | 1.000 |
| train-00000-of-00001\_2col.xlsx | 0.966 | 0.966 |
| merged\_dataset\_CSV\_\_1\_\_2col.xlsx | 0.967 | 0.967 |

#### 🚀 Semantic Relationship Scores
The scores below quantify the models' success in grouping related words (Synonyms) and distancing contrasting words (Antonyms and Separation). **FastText demonstrated superior performance in all semantic tasks.**

| Metric | W2V Score | FT Score |
| :--- | :--- | :--- |
| Synonyms | 0.297 | **0.352** |
| Antonyms | 0.243 | **0.287** |
| Separation (Contrast) | 0.053 | **0.066** |

---

### 4.2. Qualitative Analysis: Nearest Neighbors (NN)

The nearest neighbor review is crucial for identifying model robustness. FastText (FT) consistently finds minor spelling variations, proving its resilience against typos. Word2Vec (W2V), conversely, often defaults to irrelevant, highly-frequent tokens (e.g., *baktelecom\_NEG*) or seemingly random words when dealing with noisy or low-frequency terms.

| Target Word | W2V Nearest Neighbors (NN) | FT Nearest Neighbors (NN) |
| :--- | :--- | :--- |
| **yaxşı** (good) | ['nehre', 'awsome', 'çıxartmadı', 'calxalamaq', 'mərdəkanda'] | ['**yaxşıı**', '**yaxş**', 'yaxşl', 'yaxşıkı', 'uaxşı'] |
| **pis** (bad) | ['**gpon_NEG**', '**baxmaqa_NEG**', '**deyer_NEG**', 'baktelecom_NEG', 'yedi_NEG'] | ['**pisss**', '**pisə**', 'piis', 'pisti', 'pisi'] |
| **çox** (very/much) | ['**çöx**', 'lezzetlidir', 'xidmətdii', 'çpx', 'coc'] | ['**çoxçox**', '**çoxx**', 'çhox', 'çoxh', 'çoxxx'] |
| **bahalı** (expensive) | ['kantakt', 'xörəklər', 'sabuncudur', 'yaslı', 'yaxtaları'] | ['**bahalıı**', '**bahalıq**', 'bahalısı', 'pahalı', 'bahalıdı'] |
| **ucuz** (cheap) | ['sududu', 'satmalidi', 'sorbasi', 'şeytanbazardan', 'nusretden'] | ['**ucuzu**', '**ucuza**', 'ucuzda', 'ucuzdu', 'ucuzlawib'] |
| **mükəmməl** (perfect) | ['vaxdından', 'yaradilanlarin', 'carrey', 'keanu', 'netmoney'] | ['**mükəmməll**', '**mükəmməlin**', 'mükəmməlsən', 'mukəmməl', 'mükəmməldi'] |
| **dəhşət** (terrible) | ['chand', 'baori', 'yelləndi', 'çılçıraq', 'treylerdə'] | ['**dəhşəti**', '**dəhşətdü**', 'dəhşətttli', 'dəhşətiymiş', 'dəhşətə'] |
| **\<PRICE>** (placeholder) | [] | ['haunebu', 'reeeceeep', 'rinqdə', 'reytinqdə', 'üdm'] |
| **\<RATING_POS>** (placeholder) | ['qəşəg', 'əylənib', 'baga', 'proqramdır', 'kullanmalı'] | ['**\<RATING_NEG>**', '**əladıı**', 'əlaadıı', 'çookk', 'proqramdıı'] |

---

## 5. Conclusion: Model Selection

The collective evidence strongly indicates that **FastText is the more appropriate and robust embedding model** for processing Azerbaijani user-generated content.

**Justification:**

* **Superior Morphological Handling:** As an agglutinative language, Azerbaijani features many suffixes. FastText's use of character n-grams allows it to infer vectors for various morphological forms of a word (e.g., *ucuz* and *ucuzda*), a capability that Word2Vec lacks.
* **Resilience to Typos and Noise:** FastText's inherent structure provides natural robustness against common misspellings (e.g., *pisss*, *çoxx*), minimizing the impact of noise prevalent in social media data.
* **Semantic Consistency:** While Word2Vec captured some contextual links, FastText demonstrated a better overall understanding of semantic distance, as reflected by its superior scores in the Similarity and Separation metrics.
