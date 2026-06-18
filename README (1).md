# 🏆 Sports News IR Dashboard

An end-to-end **Information Retrieval & Text Mining** project for sports news articles (Football, Basketball, Tennis, F1). It combines a classic search-engine backend with machine-learning classification, all wrapped in a single interactive [Dash](https://dash.plotly.com/) web app.

Built as a final project for an Information Retrieval course.

---

## ✨ Features

The app is organized into six tabs:

| Tab | What it does |
|---|---|
| 📊 **EDA Dashboard** | Charts on sport/source distribution, headline length, body length, top keywords per sport, and a sport-filterable word cloud |
| 🗃️ **Data Explorer** | Sortable / filterable table of every article, with a click-through detail view |
| 🔍 **IR Search** | A real search engine over the corpus — switch between **TF-IDF cosine similarity**, **BM25**, and **Boolean (AND / OR / NOT)** retrieval, with highlighted snippets |
| 🤖 **AI Classifier** | Predicts **Sport** and **Sub-Topic** for any pasted headline/article using TF-IDF + LinearSVC, with softmax-normalised confidence bars and the top TF-IDF features driving each prediction |
| 📈 **Evaluation** | Held-out test-set classification reports, cross-validation scores, and an honest discussion of model limitations |
| 🕷️ **Live Scraper** | On-demand, **robots.txt-compliant** scraper for BBC Sport, Sky Sports, and CBS Sports, with live progress logging |

## 🧠 IR & ML Techniques Implemented

- **Inverted index** built from a custom regex tokenizer + NLTK stopword removal + lemmatization
- **TF-IDF cosine similarity** ranking
- **BM25** ranking (Okapi formulation, configurable k1/b)
- **Boolean retrieval** with `AND` / `OR` / `NOT` operators
- **LinearSVC** classifiers (separate models for Sport and Sub-Topic) trained on TF-IDF features (1–2 grams), evaluated with an 80/20 stratified split plus k-fold cross-validation
- Rule-based + dictionary-based **sub-topic normalisation** for noisy/raw category labels
- A polite, **robots.txt-aware** scraper with crawl-delay throttling and per-source link extraction

## 🛠️ Tech Stack

Python · Dash · Plotly · scikit-learn · NLTK · BeautifulSoup4 · Pandas · WordCloud

## 📂 Project Structure

```
.
├── app.py                          # Main Dash application (everything lives here)
├── requirements.txt                # Python dependencies
├── sports_news_cleaned (1).json    # Article data (add your own — see below)
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> NLTK stopwords/wordnet corpora are downloaded automatically on first run.

### 3. Add your data file

Place your cleaned JSON dataset in the project root. The app auto-detects it in this order:

1. The path passed via `--data`
2. `sports_news_cleaned (1).json`
3. Any `*.json` file with `sports` in the name
4. The first `*.json` file found in the folder

Each article object is expected to look roughly like:

```json
{
  "source": "BBC Sport",
  "sport": "Football",
  "headline": "...",
  "author": "...",
  "date": "2024-05-01",
  "body": "...",
  "url": "https://..."
}
```

Missing fields are filled with safe defaults, and `sub_topic` is derived automatically if not present.

### 4. Run the app

```bash
python app.py
```

Optional flags:

```bash
python app.py --data path/to/data.json --host 0.0.0.0 --port 8050
```

Then open **http://127.0.0.1:8050** in your browser.

## ⚠️ Scraper Notes

The Live Scraper tab fetches fresh articles on demand. It:
- Checks and respects each site's `robots.txt` (disallowed paths + crawl-delay)
- Identifies itself with a descriptive `User-Agent`
- Is intended for **academic/educational use only**, with modest per-run article limits

## 📋 License

This project is released under the [MIT License](LICENSE) — feel free to adapt it for your own coursework or experiments.
