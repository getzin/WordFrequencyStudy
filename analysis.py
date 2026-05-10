# analysis.py

from pathlib import Path
from collections import Counter  # used for unigram/bigram counting

import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk
import pandas as pd
import seaborn as sns
import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError

from text_stats import (
    word_lengths_sampled,
    stopword_ratio,
    type_token_ratio,
    gini_coefficient,
    top_words_normalized,
    top_bigrams_normalized,
    top_bigrams_no_stopwords,
    sentence_lengths_sampled,
    quote_per_1000_words,
    sentence_quote_ratio,
    strip_wikipedia_sections,
)

# Display-only labels for presentation (as in marimo)
CORPUS_DISPLAY = {"news": "Nachrichten", "wikipedia": "Wikipedia"}

# =============================================================================
# Matplotlib style / background (mirror marimo)
# =============================================================================

# Reset any previous style/theme changes
plt.style.use("default")
mpl.rcdefaults()

# Force white backgrounds regardless of UI theme
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)

# =============================================================================
# Data setup
# =============================================================================

# NLTK stores tokenizers/stopwords in a local data directory. Download silently if missing.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # some NLTK versions need this for sentence splitting
nltk.download("stopwords", quiet=True)

# Load news corpus (10kGNAD) --- https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv
# The file uses a semicolon separator; some rows may contain extra separators,
# so we use the python engine and keep only the first two columns.
df_news = pd.read_csv(
    "data/articles.csv",
    sep=";",
    header=None,
    names=["category", "text"],
    engine="python",
    usecols=[0, 1],
    on_bad_lines="skip",
    encoding="utf-8",
)
df_news["category"] = df_news["category"].astype(str).str.strip()
df_news["text"] = df_news["text"].astype(str).str.strip()

news_text = " ".join(df_news["text"].dropna())
print("news rows:", len(df_news), "| news chars:", len(news_text))

# Load Wikipedia corpus (cached).
# We build a small corpus by concatenating several German Wikipedia pages.
# To avoid re-downloading on every run, we cache the concatenated text in data/wiki.txt.
wikipedia.set_lang("de")
topics = [
    # Places / politics
    "Deutschland",
    "Berlin",
    "Hamburg",
    "München",
    "Köln",
    "Europa",
    "Europäische Union",
    "Politik",
    "Bundestag",
    "Bundesregierung",
    "Bundesrat (Deutschland)",
    "Föderalismus",
    "Wahlen in Deutschland",
    "Grundgesetz",
    "Menschenrechte",
    # History
    "Geschichte",
    "Deutsche Geschichte",
    "Weimarer Republik",
    "Deutsche Wiedervereinigung",
    "Kalter Krieg",
    "Französische Revolution",
    "Industrialisierung",
    # Culture / society
    "Kultur",
    "Philosophie",
    "Soziologie",
    "Psychologie",
    "Religion",
    "Bildung",
    "Literatur",
    "Musik",
    "Kunst",
    "Film",
    "Theater",
    "Architektur",
    # Science / tech
    "Naturwissenschaft",
    "Mathematik",
    "Physik",
    "Chemie",
    "Biologie",
    "Informatik",
    "Künstliche Intelligenz",
    "Maschinelles Lernen",
    "Internet",
    "Datenbank",
    "Programmierung",
    "Algorithmus",
    # Economy / environment / health
    "Wirtschaft",
    "Finanzmarkt",
    "Globalisierung",
    "Inflation",
    "Klima",
    "Klimawandel",
    "Energie",
    "Erneuerbare Energien",
    "Medizin",
]
wiki_path = Path("data/wiki.txt")

if wiki_path.exists():
    wiki_text = wiki_path.read_text(encoding="utf-8")
    wiki_loaded_from_cache = True
else:
    wiki_loaded_from_cache = False
    texts = []
    for t in topics:
        try:
            page_text = wikipedia.page(t, auto_suggest=False).content
            # Remove sections like "Literatur", "Einzelnachweise", "Weblinks", etc.
            page_text = strip_wikipedia_sections(page_text)
            texts.append(page_text)
        except (PageError, DisambiguationError):
            print("skipped:", t)
    wiki_text = " ".join(texts)
    wiki_path.write_text(wiki_text, encoding="utf-8")

print(
    "wiki chars:",
    len(wiki_text),
    "| topics:",
    len(topics),
    "| cached:",
    wiki_loaded_from_cache,
)

corpora = {"news": news_text, "wikipedia": wiki_text}


def print_h(title: str) -> None:
    """Print a small separator between hypothesis blocks."""
    print(f"\n--- {title} ---")


# =============================================================================
# Plot helper (mirror marimo: take ax and set white backgrounds)
# =============================================================================
def finalize_plot(
    ax,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rotate_x: bool = False,
):
    """Apply labels, optional x-label rotation, tight layout, and show."""
    ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if rotate_x:
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.tight_layout()
    plt.show()


# =============================================================================
# H1: Word length
# =============================================================================
print_h("H1: Word length")

rows = []
for name, text in corpora.items():
    # Sample word lengths to keep runtime reasonable on large corpora.
    for wl in word_lengths_sampled(text, max_words=200_000):
        rows.append(
            {
                "Korpus": CORPUS_DISPLAY.get(name, name),
                "Wortlänge (Zeichen)": wl,
            }
        )
df_wlen = pd.DataFrame(rows)

fig_h1, ax_h1 = plt.subplots()
sns.boxplot(data=df_wlen, x="Korpus", y="Wortlänge (Zeichen)", ax=ax_h1)
ax_h1.set_ylim(0, 25)
finalize_plot(
    ax_h1,
    title="H1: Verteilung der Wortlängen",
    xlabel="Korpus",
    ylabel="Wortlänge (Zeichen)",
)

# =============================================================================
# H2: Stopword ratio
# =============================================================================
print_h("H2: Stopword ratio")

rows = []
for name, text in corpora.items():
    rows.append(
        {
            "Korpus": CORPUS_DISPLAY.get(name, name),
            "Stopword-Anteil": stopword_ratio(text),
        }
    )
df_stop = pd.DataFrame(rows)
print(df_stop)

fig_h2, ax_h2 = plt.subplots()
sns.barplot(data=df_stop, x="Korpus", y="Stopword-Anteil", ax=ax_h2)
ax_h2.set_ylim(0, df_stop["Stopword-Anteil"].max() * 1.2)
finalize_plot(
    ax_h2,
    title="H2: Stopword-Anteil",
    xlabel="Korpus",
    ylabel="Anteil Stopwords",
    rotate_x=True,
)

# =============================================================================
# H3: Vocabulary diversity and repetition (TTR + Gini)
# =============================================================================
print_h("H3: Vocabulary diversity and repetition")

rows = []
token_sample_h3 = None

for name, text in corpora.items():
    # Simple token filter: keep alphabetic tokens only and lowercase for counting.
    words = [w.lower() for w in text.split() if w.isalpha()]
    freqs = Counter(words)

    if token_sample_h3 is None and name == "news":
        token_sample_h3 = " ".join(words[:25])

    rows.append(
        {
            "Korpus": CORPUS_DISPLAY.get(name, name),
            "TTR (unique/total)": type_token_ratio(words),  # H3a
            "Gini (frequency inequality)": gini_coefficient(freqs),  # H3b
        }
    )

df_vocab = pd.DataFrame(rows)
print(df_vocab)

if token_sample_h3 is not None:
    print("\nToken-Beispiel (Nachrichten, erste ~25 Tokens nach Filter):")
    print(token_sample_h3)

fig_h3a, ax_h3a = plt.subplots()
sns.barplot(data=df_vocab, x="Korpus", y="TTR (unique/total)", ax=ax_h3a)
finalize_plot(
    ax_h3a,
    title="H3a: Type–Token Ratio (vocabulary richness)",
    xlabel="Korpus",
    ylabel="TTR",
    rotate_x=True,
)

fig_h3b, ax_h3b = plt.subplots()
sns.barplot(data=df_vocab, x="Korpus", y="Gini (frequency inequality)", ax=ax_h3b)
finalize_plot(
    ax_h3b,
    title="H3b: Gini coefficient (word frequency concentration)",
    xlabel="Korpus",
    ylabel="Gini",
    rotate_x=True,
)

# =============================================================================
# H4: Most common individual words (unigrams)
# =============================================================================
print_h("H4: Top words (unigrams)")

rows = []
for corpus_name, text in corpora.items():
    for w, val in top_words_normalized(text, n=20):
        rows.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_name, corpus_name),
                "Wort": w,
                "Vorkommen pro 10k": val,
            }
        )
df_words = pd.DataFrame(rows)

g_h4 = sns.catplot(
    data=df_words,
    x="Vorkommen pro 10k",
    y="Wort",
    col="Korpus",
    kind="bar",
    sharex=False,
    sharey=False,
    height=5,
    aspect=1.1,
)
g_h4.set_titles("H4: Top 20 Wörter — {col_name}")
g_h4.set_axis_labels("Vorkommen pro 10.000 Wörter", "Wort")
g_h4.fig.tight_layout()
g_h4.fig.patch.set_facecolor("white")
plt.show()

# =============================================================================
# H5: Most common word pairs (bigrams)
# =============================================================================
print_h("H5: Most common word pairs (bigrams)")

# H5a: all-word bigrams
print_h("H5a: Top bigrams (all words)")

rows = []
for corpus_name, text in corpora.items():
    for (w1, w2), val in top_bigrams_normalized(text, n=20):
        rows.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_name, corpus_name),
                "Bigram": f"{w1} {w2}",
                "Vorkommen pro 100k": val,
            }
        )
df_bi_all = pd.DataFrame(rows)

g_h5a = sns.catplot(
    data=df_bi_all,
    x="Vorkommen pro 100k",
    y="Bigram",
    col="Korpus",
    kind="bar",
    sharex=False,
    sharey=False,
    height=5,
    aspect=1.1,
)
g_h5a.set_titles("H5a: Top 20 Bigrams (alle Wörter) — {col_name}")
g_h5a.set_axis_labels("Vorkommen pro 100.000 Bigrams", "Bigram")
g_h5a.fig.tight_layout()
g_h5a.fig.patch.set_facecolor("white")
plt.show()

# H5b: content-word bigrams (stopwords removed)
print_h("H5b: Top bigrams (stopwords removed)")

rows = []
for corpus_name, text in corpora.items():
    for (w1, w2), val in top_bigrams_no_stopwords(text, n=20):
        rows.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_name, corpus_name),
                "Bigram": f"{w1} {w2}",
                "Vorkommen pro 100k": val,
            }
        )
df_bi_content = pd.DataFrame(rows)

g_h5b = sns.catplot(
    data=df_bi_content,
    x="Vorkommen pro 100k",
    y="Bigram",
    col="Korpus",
    kind="bar",
    sharex=False,
    sharey=False,
    height=5,
    aspect=1.1,
)
g_h5b.set_titles("H5b: Top 20 Bigrams (Stopwords entfernt) — {col_name}")
g_h5b.set_axis_labels("Vorkommen pro 100.000 Content-Bigrams", "Bigram")
g_h5b.fig.tight_layout()
g_h5b.fig.patch.set_facecolor("white")
plt.show()

# =============================================================================
# H6: Sentence length (sampled)
# =============================================================================
print_h("H6: Sentence length")

rows = []
for name, text in corpora.items():
    for l in sentence_lengths_sampled(text, max_sentences=50_000):
        rows.append(
            {
                "Korpus": CORPUS_DISPLAY.get(name, name),
                "Satzlänge (Wörter)": l,
            }
        )
df_sent = pd.DataFrame(rows)

fig_h6, ax_h6 = plt.subplots()
sns.boxplot(data=df_sent, x="Korpus", y="Satzlänge (Wörter)", ax=ax_h6)
ax_h6.set_ylim(0, 60)
finalize_plot(
    ax_h6,
    title="H6: Verteilung der Satzlängen",
    xlabel="Korpus",
    ylabel="Satzlänge (Wörter)",
)

# =============================================================================
# H7: Quotes
# =============================================================================
print_h("H7: Quotes")

rows = []
for name, text in corpora.items():
    rows.append(
        {
            "Korpus": CORPUS_DISPLAY.get(name, name),
            "Anführungszeichen pro 1000 Wörter": quote_per_1000_words(text),  # H7a
            "Sätze mit Anführungszeichen (%)": sentence_quote_ratio(text) * 100,  # H7b
        }
    )
df_quotes = pd.DataFrame(rows)
print(df_quotes)

fig_h7a, ax_h7a = plt.subplots()
sns.barplot(data=df_quotes, x="Korpus", y="Anführungszeichen pro 1000 Wörter", ax=ax_h7a)
finalize_plot(
    ax_h7a,
    title="H7a: Anführungszeichen pro 1.000 Wörter",
    xlabel="Korpus",
    ylabel="Anführungszeichen pro 1.000 Wörter",
    rotate_x=True,
)

fig_h7b, ax_h7b = plt.subplots()
sns.barplot(data=df_quotes, x="Korpus", y="Sätze mit Anführungszeichen (%)", ax=ax_h7b)
ax_h7b.set_ylim(0, df_quotes["Sätze mit Anführungszeichen (%)"].max() * 1.2)
finalize_plot(
    ax_h7b,
    title="H7b: Anteil der Sätze mit Anführungszeichen",
    xlabel="Korpus",
    ylabel="Sätze mit Anführungszeichen (%)",
    rotate_x=True,
)

# End of analysis.py
