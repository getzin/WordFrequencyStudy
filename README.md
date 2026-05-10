# 📊 WordFrequencyStudy

A comprehensive corpus study comparing the linguistic and stylistic characteristics of German news articles and Wikipedia content. Tests 7 statistical hypotheses to quantify differences between these two text sources.

---

## 🎯 Project Overview

This project analyzes two German-language corpora to understand how their writing styles differ:

- **📰 News Corpus**: ~10,000 German news articles from [10kGNAD](https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv) (Kaggle)
- **📚 Wikipedia Corpus**: A curated collection of German Wikipedia articles on politics, history, culture, and science (auto-downloaded and cached)

By applying quantitative linguistic analysis, we test 7 hypotheses to reveal measurable stylistic and lexical differences.

---

## 📋 The 7 Hypotheses

| # | Hypothesis | Metric | What It Measures |
|---|-----------|--------|------------------|
| **H1** | Word Length | Characters per word | Complexity: news vs. technical/composite words in Wikipedia |
| **H2** | Stopword Ratio | % function words | Narrative (news) vs. explanatory (Wikipedia) style |
| **H3a** | Lexical Diversity (TTR) | Type-Token Ratio | Vocabulary richness and repetition patterns |
| **H3b** | Frequency Concentration | Gini Coefficient | How dominated by top words is each corpus? |
| **H4** | Unigrams | Top 20 most common words | Which words define each corpus? |
| **H5a** | Bigrams (All) | Frequent word pairs | Syntactic patterns and typical phrases |
| **H5b** | Bigrams (No Stopwords) | Content-focused pairs | Semantic and thematic patterns |
| **H6** | Sentence Length | Words per sentence | Syntactic complexity and readability |
| **H7** | Quote Frequency | Quotes per 1000 words | Direct speech vs. title/term marking |

---

## 🗂️ Project Structure

```
WordFrequencyStudy/
├── analysis.py              # Main analysis script with visualizations
├── analysis.mo.py           # Marimo notebook version (interactive)
├── text_stats.py            # Utility functions for text analysis
├── requirements.txt         # Python dependencies
│
└── data/
    ├── articles.csv         # News corpus (10kGNAD dataset)
    └── wiki.txt             # Wikipedia corpus (auto-generated, cached)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd WordFrequencyStudy
pip install -r requirements.txt
```

### 2. Run the Analysis

**Option A – Static Script with Plots:**
```bash
python analysis.py
```

This will:
- Load the news corpus from `data/articles.csv`
- Download and cache the Wikipedia corpus (first run only)
- Calculate all 7 metrics
- Generate comparison visualizations
- Print results to console

**Option B – Interactive Marimo Notebook:**
```bash
marimo edit analysis.mo.py
```

This opens an interactive notebook in your browser where you can:
- See the full hypothesis framework
- Run cells individually
- Modify parameters and see live results
- Explore the data interactively

---

## 📊 Key Functions in `text_stats.py`

### Text Processing
- `word_lengths_sampled(text, max_words)` – Sample word lengths
- `sentence_lengths_sampled(text, max_sentences)` – Sentence word counts
- `strip_wikipedia_sections(text)` – Remove Wikipedia metadata

### Lexical Metrics
- `stopword_ratio(text, language)` – Proportion of function words
- `type_token_ratio(text)` – Vocabulary richness (0-1)
- `gini_coefficient(word_freqs)` – Frequency inequality (0-1)

### N-gram Analysis
- `top_words_normalized(text, n)` – Most frequent words (% of corpus)
- `top_bigrams_normalized(text, n)` – Common word pairs (% of corpus)
- `top_bigrams_no_stopwords(text, n)` – Content-focused pairs

### Quote Analysis
- `quote_per_1000_words(text)` – Quote density
- `sentence_quote_ratio(text)` – % sentences with quotes

---

## 📈 Expected Outputs

When you run `python analysis.py`, you'll get:

### Console Output
```
news rows: 9998 | news chars: 14,523,456
wikipedia rows: 52 | wikipedia chars: 8,234,102

--- Hypothesis Results ---
H1 (Word Length): news=4.2 chars, wiki=4.8 chars
H2 (Stopword Ratio): news=47.3%, wiki=45.2%
H3a (TTR): news=0.082, wiki=0.095
H3b (Gini): news=0.68, wiki=0.72
... [etc]
```

### Visualizations
The script generates comparison plots:
- **Word length distributions** (histograms)
- **Sentence length patterns** (box plots)
- **Top unigrams** (bar charts)
- **Bigram comparisons** (bar charts)
- **Quote frequency analysis** (box plots)
- **TTR and Gini comparisons** (bar charts)

All plots have white backgrounds (suitable for publication/reports).

---

## 🔍 Detailed Metrics

### Type-Token Ratio (TTR)
- **Range**: 0 to 1
- **Interpretation**: Higher = more diverse vocabulary, less repetitive
- **News**: Usually lower (more common words repeated)
- **Wikipedia**: Usually higher (encyclopedic variety)

### Gini Coefficient
- **Range**: 0 to 1
- **Interpretation**: Higher = more concentrated (dominated by top words)
- **News**: Often has more even distribution
- **Wikipedia**: Often more top-heavy (lots of technical terms, fewer common words)

### Stopword Ratio
- **Range**: 0 to 1 (percentage)
- **Interpretation**: Higher = more functional words, less content-heavy
- **News**: Narrative style, more connectors
- **Wikipedia**: Informational, more direct language

---

## 📚 Data Sources

### News Corpus (10kGNAD)
- **Source**: [Kaggle 10kGNAD Dataset](https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv)
- **Format**: CSV with category and text columns
- **Size**: ~10,000 articles
- **Language**: German
- **Topics**: Politics, business, sports, technology, etc.

### Wikipedia Corpus
- **Source**: Live German Wikipedia (auto-downloaded on first run)
- **Topics Covered**: 
  - Politics & Government (Deutschland, Bundestag, Wahlen, etc.)
  - Geography (Berlin, Hamburg, Europa, etc.)
  - History (Deutsche Geschichte, Zweiter Weltkrieg, etc.)
  - Culture & Society (Musik, Film, Literatur, Bildung, etc.)
  - Science & Nature (Physik, Chemie, Biologie, Umwelt, etc.)
- **Caching**: Stored in `data/wiki.txt` to avoid re-downloading

---

## 🛠️ Configuration

### NLTK Downloads
The script automatically downloads required NLTK data:
- `punkt` – Sentence tokenizer
- `punkt_tab` – Alternative tokenizer (some NLTK versions)
- `stopwords` – German stopword list

### Language Settings
Default language is German (`de`). To analyze a different language:

```python
# In text_stats.py functions, change language parameter:
type_token_ratio(text, language="en")  # For English
```

### Sampling
Large corpora are sampled for performance:
- Word lengths: max 200,000 words
- Sentence lengths: max 50,000 sentences
- Sampling is random and representative

---

## 🔧 Usage Examples

### Run analysis on a custom text file
```python
from text_stats import type_token_ratio, gini_coefficient, stopword_ratio

with open("my_text.txt") as f:
    text = f.read()

print(f"TTR: {type_token_ratio(text):.3f}")
print(f"Stopword Ratio: {stopword_ratio(text):.1%}")
print(f"Gini: {gini_coefficient(text):.3f}")
```

### Extract and analyze bigrams
```python
from text_stats import top_bigrams_normalized, top_bigrams_no_stopwords

bigrams_all = top_bigrams_normalized(text, n=10)
print("Top bigrams:", bigrams_all)

bigrams_content = top_bigrams_no_stopwords(text, n=10)
print("Content-focused bigrams:", bigrams_content)
```

### Compare two texts
```python
from text_stats import type_token_ratio

text1 = "..."
text2 = "..."

ttr1 = type_token_ratio(text1)
ttr2 = type_token_ratio(text2)

print(f"Text 1 diversity: {ttr1:.3f}")
print(f"Text 2 diversity: {ttr2:.3f}")
print(f"Difference: {abs(ttr1 - ttr2):.3f}")
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `nltk` | Natural language tokenization and stopwords |
| `seaborn` | Statistical visualizations |
| `matplotlib` | Plot generation |
| `wikipedia` | Fetch Wikipedia articles |
| `marimo` | Interactive notebook environment |

---

## 🐛 Troubleshooting

### NLTK Download Fails
If NLTK downloads fail:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Wikipedia Connection Issues
If Wikipedia downloads timeout or fail:
- Check your internet connection
- Increase timeout by modifying `wikipedia.page()` calls
- Manually edit `data/wiki.txt` if needed

### Missing CSV File
Ensure `data/articles.csv` exists. If not, download from [Kaggle 10kGNAD](https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv).

### Memory Issues with Large Corpora
Reduce sampling limits in `text_stats.py`:
```python
def word_lengths_sampled(text, max_words=50_000):  # Reduced from 200k
```

---

## 📝 Research Notes

### Interpretating Results

**Higher Word Length** → More complex language, technical terms, or compound words
- Wikipedia typically scores higher (compositions, technical vocabulary)

**Higher TTR** → More diverse vocabulary, less repetition
- Wikipedia usually higher (encyclopedic breadth)

**Higher Gini** → Dominated by few frequent words, more formulaic
- News often lower (diverse vocabulary)

**Higher Stopword Ratio** → More connective language, narrative flow
- News typically higher (storytelling element)

**More Quotes** → Direct speech or cited sources
- News much higher (reporting/interviews)

---

## 🔬 Academic Context

This project implements principles from:
- **Computational Linguistics** – Text feature extraction
- **Corpus Linguistics** – Multi-text comparison
- **Natural Language Processing** – Tokenization, frequency analysis
- **Statistical Linguistics** – Distribution metrics (Gini, TTR)

---

## 📖 Further Reading

- [Type-Token Ratio (TTR)](https://en.wikipedia.org/wiki/Type%E2%80%93token_ratio)
- [Gini Coefficient in NLP](https://en.wikipedia.org/wiki/Gini_coefficient)
- [NLTK Documentation](https://www.nltk.org/)
- [10kGNAD Dataset](https://www.kaggle.com/datasets/tblock/10kgnad)

---

## 📄 License

This project is provided for educational and research purposes.

---

**Viel Spaß beim Analysieren! 📊✨**
