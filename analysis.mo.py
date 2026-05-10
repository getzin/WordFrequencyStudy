# analysis.mo.py
# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    # Ensure the project root is searched before site-packages.
    # This prevents accidental imports from pip packages with the same name.
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return (Path,)


@app.cell
def _(mo):
    mo.md(r"""
    # Nachrichten vs. Wikipedia -- Korpusstudie

    Wir vergleichen zwei deutschsprachige Korpora:

    - **Nachrichten**: 10kGNAD (Kaggle), ~10k Nachrichtenartikel (liegt in `data/articles.csv`) --- https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv
    - **Wikipedia**: ein kleines, eigenes Korpus aus zusammengefügten deutschsprachigen Wikipedia-Themen<br>
    (Wird beim ersten Run aus Wikipedia geladen und gecacht in `data/wiki.txt`)

    Ziel: Mehrere Hypothesen testen, um stilistische Unterschiede sichtbar zu machen.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 Hypothesen

    **Leitfrage:** Worin unterscheiden sich Nachrichten und Wikipedia stilistisch – messbar mit einfachen, transparenten Kennzahlen?

    Dafür testen wir sieben Hypothesen:

    - **H1 — Wortlänge:** Die durchschnittliche Wortlänge unterscheidet sich zwischen den Korpora,
      z. B. durch mehr Komposita und Fachbegriffe in Wikipedia.

    - **H2 — Stopword-Anteil:** Der Anteil an Stopwords unterscheidet sich und kann auf
      narrativen (Nachrichten) vs. erklärenden Stil (Wikipedia) hinweisen.

    - **H3 — Wortschatz (Vielfalt & Wiederholung):**
      - **H3a (TTR):** Die lexikalische Vielfalt unterscheidet sich messbar über die
          **Type–Token Ratio (TTR)** *(höher = vielfältiger Wortschatz)*.
      - **H3b (Gini):** Die Konzentration der Worthäufigkeiten unterscheidet sich messbar
          über den **Gini-Koeffizienten** *(höher = stärkere Konzentration auf wenige Wörter)*.

    - **H4 — Häufigste Wörter (Unigrams):** Die dominanten Wörter unterscheiden sich
      in Rangfolge und relativer Häufigkeit (normalisiert).

    - **H5 — Wortpaare (Bigrams):**
      - **H5a:** Häufige Bigrams über alle Wörter spiegeln typische **syntaktische Muster** wider.
      - **H5b:** Bigrams ohne Stopwords machen eher **inhaltliche und semantische Muster** sichtbar.

    - **H6 — Satzlänge:** Die Satzlängenverteilungen unterscheiden sich und geben Hinweise
      auf unterschiedlich komplexe Satzstrukturen.

    - **H7 — Anführungszeichen:** Die Nutzung von Anführungszeichen unterscheidet sich
      zwischen den Korpora, etwa für direkte Rede versus Begriffs- oder Titelmarkierung.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup & Imports

    Wir verwenden:

    - `pandas` zum Laden und Umformen der Daten
    - `nltk` für Tokenisierung + Stopwords
    - `seaborn/matplotlib` für Plots
    - das Python-Paket `wikipedia` zum Abrufen von Seiten (anschließend Cache)
    - Hilfsfunktionen liegen in **`text_stats.py`**
    """)
    return


@app.cell
def _():
    from collections import Counter

    import matplotlib.pyplot as plt
    import nltk
    import pandas as pd
    import seaborn as sns
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError

    from text_stats import (
        gini_coefficient,
        quote_per_1000_words,
        sentence_lengths_sampled,
        sentence_quote_ratio,
        stopword_ratio,
        strip_wikipedia_sections,
        top_bigrams_no_stopwords,
        top_bigrams_normalized,
        top_words_normalized,
        type_token_ratio,
        word_lengths_sampled,
    )

    # Display-only labels for presentation
    CORPUS_DISPLAY = {"news": "Nachrichten", "wikipedia": "Wikipedia"}
    return (
        CORPUS_DISPLAY,
        Counter,
        DisambiguationError,
        PageError,
        gini_coefficient,
        nltk,
        pd,
        plt,
        quote_per_1000_words,
        sentence_lengths_sampled,
        sentence_quote_ratio,
        sns,
        stopword_ratio,
        strip_wikipedia_sections,
        top_bigrams_no_stopwords,
        top_bigrams_normalized,
        top_words_normalized,
        type_token_ratio,
        wikipedia,
        word_lengths_sampled,
    )


@app.cell
def _(plt):
    import matplotlib as mpl

    # Reset any previous style/theme changes (helps when switching marimo themes)
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
    return


@app.cell
def _(nltk):
    # NLTK resources: download silently if missing
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)  # some NLTK versions need this
    nltk.download("stopwords", quiet=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Datenquellen

    ### Nachrichten-Korpus (10kGNAD -- Ten Thousand German News Articles Dataset)

    Download link @ kaggle:
    https://www.kaggle.com/datasets/tblock/10kgnad?select=articles.csv<br>
    Wird geladen aus: `data/articles.csv`<br>
    Jede Zeile ist ein Artikel und besteht aus 2 columns: *(category, text)*.<br>

    ### Wikipedia-Korpus

    Wir holen eine Liste von Themen und fügen sie zu einem großen Text zusammen.<br>
    Das File wird gecached unter: `data/wiki.txt`<br>
    Im Gegensatz zu den Nachrichten, sind die Wikipedia in einem langen String konkateniert.
    """)
    return


@app.cell
def _(pd):
    # Load news corpus (10kGNAD).
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
    return df_news, news_text


@app.cell
def _(df_news, mo, news_text):
    snippet_news = news_text[:300].replace("\n", " ")

    mo.vstack(
        [
            mo.md(
                r"""
    ### Nachrichten-Datensatz (Kurze Übersicht)

    Hier exemplarisch die ersten 3 Zeilen:
    """
            ),
            mo.ui.table(df_news.head(3)),
            mo.md("&nbsp;\n&nbsp;"), #vertical space
            mo.md("**Erste 300 Zeichen:**"),
            mo.md(f"```text\n{snippet_news}\n```"),
        ]
    )
    return


@app.cell
def _(
    DisambiguationError,
    PageError,
    Path,
    strip_wikipedia_sections,
    wikipedia,
):
    # Load Wikipedia corpus (cached).
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
        wiki_first_page_sample = None
    else:
        wiki_loaded_from_cache = False
        texts = []
        wiki_first_page_sample = None

        for i, topic in enumerate(topics):
            try:
                page_text = wikipedia.page(topic, auto_suggest=False).content
                page_text = strip_wikipedia_sections(page_text)
                texts.append(page_text)

                if i == 0:
                    wiki_first_page_sample = page_text[:300].replace("\n", " ")
            except (PageError, DisambiguationError):
                print("skipped:", topic)

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
    return topics, wiki_first_page_sample, wiki_loaded_from_cache, wiki_text


@app.cell
def _(mo, topics, wiki_first_page_sample, wiki_loaded_from_cache, wiki_text):
    # Show ONE snippet only (from the aggregated corpus), plus the topic list.
    snippet_wiki = wiki_text[:300].replace("\n", " ")

    # Render topics nicely (compact, but readable)
    topics_md = "\n".join([f"- {t}" for t in topics])

    blocks_wiki = [
        mo.md(
            r"""
    ### Wikipedia-Datensatz (Kurze Übersicht)

    Wir zeigen:
    - die verwendeten Themen (Topics)
    - ob der Text aus dem Cache geladen wurde
    - einen kurzen Ausschnitt aus dem **aggregierten** Wikipedia-Korpus
    """
        ),
        mo.md(f"- **Themen (Anzahl):** {len(topics)}"),
        mo.md(f"- **Aus Cache geladen:** `{wiki_loaded_from_cache}`"),
        mo.md("**Themenliste:**"),
        mo.md(topics_md),
        mo.md("**Wikipedia-Ausschnitt (erste 300 Zeichen, aggregierter Korpus):**"),
        mo.md(f"```text\n{snippet_wiki}\n```"),
    ]

    # Optional: keep the first-page sample ONLY if you want a *different* snippet.
    # If you prefer, you can show it with a different label and only when it differs.
    if wiki_first_page_sample is not None and wiki_first_page_sample.strip() != snippet_wiki.strip():
        blocks_wiki.append(
            mo.md(
                "**Zusatzbeispiel: erste Seite nach Entfernen von Abschnitten (erste 300 Zeichen):**"
            )
        )
        blocks_wiki.append(mo.md(f"```text\n{wiki_first_page_sample}\n```"))

    mo.vstack(blocks_wiki)
    return


@app.cell
def _(news_text, wiki_text):
    # concatenate both corpora

    corpora = {"news": news_text, "wikipedia": wiki_text}
    return (corpora,)


@app.function
# small helper function

def print_h(title: str) -> None:
    """Small console separator between hypothesis blocks."""
    print(f"\n--- {title} ---")


@app.function
def finalize_plot(
    ax,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rotate_x: bool = False,
):
    """Apply labels, optional x-label rotation, tight layout, and return the figure."""
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
    return fig


@app.cell
def _(mo):
    mo.md(r"""
    ## H1 — Wortlänge

    **Hypothese:** Wikipedia verwendet längere Wörter (mehr Komposita/technische Begriffe).
    Wir vergleichen die Verteilung der **Wortlänge in Zeichen** (für Performance gesampelt).
    """)
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, plt, sns, word_lengths_sampled):
    print_h("H1: Word length")

    rows_h1 = []
    for corpus_label_h1, corpus_text_h1 in corpora.items():
        for wl_h1 in word_lengths_sampled(corpus_text_h1, max_words=200_000):
            rows_h1.append(
                {
                    "Korpus": CORPUS_DISPLAY.get(corpus_label_h1, corpus_label_h1),
                    "Wortlänge (Zeichen)": wl_h1,
                }
            )
    df_wlen = pd.DataFrame(rows_h1)

    fig_h1, ax_h1 = plt.subplots()
    sns.boxplot(data=df_wlen, x="Korpus", y="Wortlänge (Zeichen)", ax=ax_h1)
    ax_h1.set_ylim(0, 25)

    fig_h1 = finalize_plot(
        ax_h1,
        title="H1: Verteilung der Wortlängen",
        xlabel="Korpus",
        ylabel="Wortlänge (Zeichen)",
    )

    fig_h1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H2 — Stopword-Anteil

    **Hypothese:** Nachrichten verwenden mehr Stopwords (Funktionswörter) – ein Hinweis auf einen narrativeren/konversationelleren Stil.

    Wir berechnen:
    **stopword_ratio = (# Stopwords) / (# alphabetische Wörter)**.
    """)
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, plt, sns, stopword_ratio):
    print_h("H2: Stopword ratio")

    rows_h2 = []
    for corpus_label_h2, corpus_text_h2 in corpora.items():
        rows_h2.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_label_h2, corpus_label_h2),
                "Stopword-Anteil": stopword_ratio(corpus_text_h2),
            }
        )
    df_stop = pd.DataFrame(rows_h2)

    fig_h2, ax_h2 = plt.subplots()
    sns.barplot(data=df_stop, x="Korpus", y="Stopword-Anteil", ax=ax_h2)
    ax_h2.set_ylim(0, df_stop["Stopword-Anteil"].max() * 1.2)

    fig_h2 = finalize_plot(
        ax_h2,
        title="H2: Stopword-Anteil",
        xlabel="Korpus",
        ylabel="Anteil Stopwords",
        rotate_x=True,
    )

    fig_h2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H3 — Wortschatz: Vielfalt und Wiederholung

    Zwei ergänzende Metriken:

    ### H3a: Type–Token Ratio (TTR)
    **TTR = unique_words / total_words**
    Höhere TTR ⇒ mehr lexikalische Vielfalt.
    (*Hinweis:* TTR hängt von der Korpusgröße ab, ist aber für Vergleiche trotzdem hilfreich.)

    ### H3b: Gini coefficient auf Worthäufigkeiten
    Misst, wie stark die Häufigkeitsverteilung konzentriert ist.
    Höherer Gini ⇒ wenige Wörter dominieren (mehr Wiederholung/Konzentration).
    """)
    return


@app.cell
def _(
    CORPUS_DISPLAY,
    Counter,
    corpora,
    gini_coefficient,
    mo,
    pd,
    plt,
    sns,
    type_token_ratio,
):
    print_h("H3: Vocabulary diversity and repetition")

    rows_h3 = []
    token_sample_h3 = None

    for corpus_label_h3, corpus_text_h3 in corpora.items():
        words_h3 = [w.lower() for w in corpus_text_h3.split() if w.isalpha()]
        freqs_h3 = Counter(words_h3)

        if token_sample_h3 is None and corpus_label_h3 == "news":
            token_sample_h3 = " ".join(words_h3[:25])

        rows_h3.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_label_h3, corpus_label_h3),
                "TTR (unique/total)": type_token_ratio(words_h3),
                "Gini (frequency inequality)": gini_coefficient(freqs_h3),
            }
        )

    df_vocab = pd.DataFrame(rows_h3)

    fig_h3a, ax_h3a = plt.subplots()
    sns.barplot(data=df_vocab, x="Korpus", y="TTR (unique/total)", ax=ax_h3a)
    fig_h3a = finalize_plot(
        ax_h3a,
        title="H3a: Type–Token Ratio (vocabulary richness)",
        xlabel="Korpus",
        ylabel="TTR",
        rotate_x=True,
    )

    fig_h3b, ax_h3b = plt.subplots()
    sns.barplot(data=df_vocab, x="Korpus", y="Gini (frequency inequality)", ax=ax_h3b)
    fig_h3b = finalize_plot(
        ax_h3b,
        title="H3b: Gini coefficient (word frequency concentration)",
        xlabel="Korpus",
        ylabel="Gini",
        rotate_x=True,
    )

    plots_h3 = mo.vstack([fig_h3a, fig_h3b])

    if token_sample_h3 is None:
        rendered_h3 = plots_h3
    else:
        token_block_h3 = mo.vstack(
            [
                mo.md("**Token-Beispiel (Nachrichten, erste ~25 Tokens nach Filter):**"),
                mo.md(f"```text\n{token_sample_h3}\n```"),
            ]
        )
        rendered_h3 = mo.vstack([token_block_h3, plots_h3])

    rendered_h3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H4 — Häufigste Wörter (Unigrams)

    Wir vergleichen die **Top 20** der häufigsten Wörter, normalisiert auf:

    **Vorkommen pro 10.000 Wörter**.

    So sieht man, welcher Wortschatz die jeweiligen Korpora dominiert.
    """)
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, sns, top_words_normalized):
    print_h("H4: Top words (unigrams)")

    rows_h4 = []
    for corpus_label_h4, corpus_text_h4 in corpora.items():
        for w_h4, val_h4 in top_words_normalized(corpus_text_h4, n=20):
            rows_h4.append(
                {
                    "Korpus": CORPUS_DISPLAY.get(corpus_label_h4, corpus_label_h4),
                    "Wort": w_h4,
                    "Vorkommen pro 10k": val_h4,
                }
            )
    df_words = pd.DataFrame(rows_h4)

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

    g_h4.fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H5 — Häufigste Wortpaare (Bigrams)

    Ein **Bigram** bedeutet hier: **zwei aufeinanderfolgende Wörter**.

    Wir vergleichen:
    - **H5a:** Bigrams über alle Wörter (oft syntaktische Muster, stark von Stopwords geprägt)
    - **H5b:** Bigrams über Inhaltswörter (Stopwords entfernt), eher semantisch

    Counts sind normalisiert auf: **Vorkommen pro 100.000 Bigrams**.
    """)
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, sns, top_bigrams_normalized):
    print_h("H5a: Top bigrams (all words)")

    rows_h5a = []
    for corpus_label_h5a, corpus_text_h5a in corpora.items():
        for (w1_h5a, w2_h5a), val_h5a in top_bigrams_normalized(corpus_text_h5a, n=20):
            rows_h5a.append(
                {
                    "Korpus": CORPUS_DISPLAY.get(corpus_label_h5a, corpus_label_h5a),
                    "Bigram": f"{w1_h5a} {w2_h5a}",
                    "Vorkommen pro 100k": val_h5a,
                }
            )
    df_bi_all = pd.DataFrame(rows_h5a)

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

    g_h5a.fig
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, sns, top_bigrams_no_stopwords):
    print_h("H5b: Top bigrams (stopwords removed)")

    rows_h5b = []
    for corpus_label_h5b, corpus_text_h5b in corpora.items():
        for (w1_h5b, w2_h5b), val_h5b in top_bigrams_no_stopwords(
            corpus_text_h5b, n=20
        ):
            rows_h5b.append(
                {
                    "Korpus": CORPUS_DISPLAY.get(corpus_label_h5b, corpus_label_h5b),
                    "Bigram": f"{w1_h5b} {w2_h5b}",
                    "Vorkommen pro 100k": val_h5b,
                }
            )
    df_bi_content = pd.DataFrame(rows_h5b)

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

    g_h5b.fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H6 — Satzlänge

    **Hypothese:** Wikipedia verwendet längere/komplexere Sätze.

    Wir sampeln bis zu **50.000 Sätze** pro Korpus und vergleichen die Verteilungen.
    """)
    return


@app.cell
def _(CORPUS_DISPLAY, corpora, pd, plt, sentence_lengths_sampled, sns):
    print_h("H6: Sentence length")

    rows_h6 = []
    for corpus_label_h6, corpus_text_h6 in corpora.items():
        for sent_len_h6 in sentence_lengths_sampled(
            corpus_text_h6, max_sentences=50_000
        ):
            rows_h6.append(
                {
                    "Korpus": CORPUS_DISPLAY.get(corpus_label_h6, corpus_label_h6),
                    "Satzlänge (Wörter)": sent_len_h6,
                }
            )
    df_sent = pd.DataFrame(rows_h6)

    fig_h6, ax_h6 = plt.subplots()
    sns.boxplot(data=df_sent, x="Korpus", y="Satzlänge (Wörter)", ax=ax_h6)
    ax_h6.set_ylim(0, 60)

    fig_h6 = finalize_plot(
        ax_h6,
        title="H6: Verteilung der Satzlängen",
        xlabel="Korpus",
        ylabel="Satzlänge (Wörter)",
    )

    fig_h6
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H7 — Anführungszeichen

    **Hypothese:** Die Nutzung von Anführungszeichen unterscheidet sich zwischen Nachrichten und Wikipedia.

    Wir betrachten zwei Maße:
    - **H7a:** Anzahl der Anführungszeichen-Zeichen pro 1.000 Wörter
    - **H7b:** Anteil der Sätze, die mindestens ein Anführungszeichen enthalten

    Diese Maße erfassen sowohl direkte Rede als auch andere Verwendungen
    (z. B. Titel- oder Begriffsmarkierungen).
    """)
    return


@app.cell
def _(
    CORPUS_DISPLAY,
    corpora,
    mo,
    pd,
    plt,
    quote_per_1000_words,
    sentence_quote_ratio,
    sns,
):
    print_h("H7: Quotes")

    rows_h7 = []
    for corpus_label_h7, corpus_text_h7 in corpora.items():
        rows_h7.append(
            {
                "Korpus": CORPUS_DISPLAY.get(corpus_label_h7, corpus_label_h7),
                "Anführungszeichen pro 1000 Wörter": quote_per_1000_words(corpus_text_h7),
                "Sätze mit Anführungszeichen (%)": sentence_quote_ratio(corpus_text_h7)
                * 100,
            }
        )
    df_quotes = pd.DataFrame(rows_h7)

    fig_h7a, ax_h7a = plt.subplots()
    sns.barplot(
        data=df_quotes, x="Korpus", y="Anführungszeichen pro 1000 Wörter", ax=ax_h7a
    )
    fig_h7a = finalize_plot(
        ax_h7a,
        title="H7a: Anführungszeichen pro 1.000 Wörter",
        xlabel="Korpus",
        ylabel="Anführungszeichen pro 1.000 Wörter",
        rotate_x=True,
    )

    fig_h7b, ax_h7b = plt.subplots()
    sns.barplot(
        data=df_quotes, x="Korpus", y="Sätze mit Anführungszeichen (%)", ax=ax_h7b
    )
    ax_h7b.set_ylim(0, df_quotes["Sätze mit Anführungszeichen (%)"].max() * 1.2)
    fig_h7b = finalize_plot(
        ax_h7b,
        title="H7b: Anteil der Sätze mit Anführungszeichen",
        xlabel="Korpus",
        ylabel="Sätze mit Anführungszeichen (%)",
        rotate_x=True,
    )

    mo.vstack([fig_h7a, fig_h7b])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fazit

    Bereits einfache Korpusmetriken zeigen **klare Stilunterschiede** zwischen **Nachrichten** und **Wikipedia**.

    ### Kernergebnisse

    - **Wortwahl:** Wikipedia nutzt häufiger **längere Wörter** und Fachbegriffe, Nachrichten eher **kürzere**.
    - **Stopwords & Verteilung:** Nachrichten haben **mehr Stopwords** und **stärker konzentrierte** Wortverteilungen (Gini ↑);
      Wikipedia ist **lexikalisch vielfältiger** (TTR ↑).
    - **Phrasen:** Nachrichten wirken **ereignis- und zahlenorientiert**, Wikipedia **thematisch und fachlich**.
    - **Satzbau:** Wikipedia verwendet **längere/variablere** Sätze, Nachrichten **kürzere/direktere**.
    - **Anführungszeichen:** In Wikipedia häufiger (Begriffe, Titel, Zitate).

    ### Zusammenfassung

    - **Nachrichten:** kompakt, narrativ, ereignisorientiert
    - **Wikipedia:** erklärend, strukturiert, wissensorientiert

    ### Einschränkungen

    - Wikipedia-Themen **manuell gewählt** (keine Zufallsstichprobe).
    - **TTR** größenabhängig → nur **vergleichend** interpretieren.
    - Anführungszeichen haben **mehrere Funktionen**.
    """)
    return


if __name__ == "__main__":
    app.run()
