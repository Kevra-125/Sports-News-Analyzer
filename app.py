import json, re, math, io, base64, warnings, argparse, os, time, threading
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              precision_score, recall_score)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  CLI — configurable data file path
# =============================================================================
parser = argparse.ArgumentParser(description='Sports IR Dashboard')
parser.add_argument('--data', type=str, default=None,
                    help='Path to cleaned JSON data file. Auto-detects if omitted.')
parser.add_argument('--port', type=int, default=8050)
parser.add_argument('--host', type=str, default='127.0.0.1')
try:
    args = parser.parse_args()
except SystemExit:
    class _Args:
        data = None; port = 8050; host = '127.0.0.1'
    args = _Args()


def find_data_file(explicit=None):
    """Find the JSON data file — explicit path > common names > any JSON."""
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = [
        'sports_news_cleaned (1).json'
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    jsons = [f for f in os.listdir('.') if f.endswith('.json') and 'sports' in f.lower()]
    if jsons:
        return jsons[0]
    jsons = [f for f in os.listdir('.') if f.endswith('.json')]
    if jsons:
        return jsons[0]
    raise FileNotFoundError(
        'No JSON data file found. Use --data path/to/file.json '
        'or place sports_news_cleaned.json in the same folder as app.py'
    )

DATA_FILE = find_data_file(args.data)
print(f'[app] Using data file: {DATA_FILE}')


# =============================================================================
# 1.  LOAD & PREPARE DATA
# =============================================================================

# FIX 2 — Extended subtopic map: covers numeric IDs, slugs, and common
# data-quality artifacts that fall through the original map as raw strings.
SUBTOPIC_MAP = {
    # Football competitions
    'premier league': 'Premier League', 'epl': 'Premier League',
    'champions league': 'Champions League', 'ucl': 'Champions League',
    'europa league': 'Europa League', 'uel': 'Europa League',
    'conference league': 'Conference League', 'uecl': 'Conference League',
    'fa cup': 'FA Cup', 'facup': 'FA Cup', 'fa-cup': 'FA Cup',
    'carabao cup': 'Carabao Cup', 'league cup': 'Carabao Cup',
    'la liga': 'La Liga', 'laliga': 'La Liga', 'primera division': 'La Liga',
    'bundesliga': 'Bundesliga', '2 bundesliga': 'Bundesliga 2',
    'serie a': 'Serie A', 'serie-a': 'Serie A',
    'ligue 1': 'Ligue 1', 'ligue1': 'Ligue 1',
    'eredivisie': 'Eredivisie',
    'mls': 'MLS',
    'world cup': 'World Cup', 'fifa world cup': 'World Cup', 'wc': 'World Cup',
    'euros': 'Euros', 'euro 2024': 'Euros', 'euro': 'Euros',
    'nations league': 'Nations League',
    # Football topics
    'transfer': 'Transfers & Rumours', 'transfers': 'Transfers & Rumours',
    'transfer news': 'Transfers & Rumours', 'rumour': 'Transfers & Rumours',
    'rumours': 'Transfers & Rumours', 'signings': 'Transfers & Rumours',
    'international': 'International Football',
    'injury': 'Injuries & Fitness', 'injuries': 'Injuries & Fitness',
    'fitness': 'Injuries & Fitness', 'injury news': 'Injuries & Fitness',
    'manager': 'Managers & Tactics', 'tactics': 'Managers & Tactics',
    'sacking': 'Managers & Tactics', 'appointment': 'Managers & Tactics',
    'match report': 'Match Report', 'matchreport': 'Match Report',
    'preview': 'Match Preview', 'match preview': 'Match Preview',
    'analysis': 'Analysis & Opinion', 'opinion': 'Analysis & Opinion',
    'award': 'Awards & Records', 'record': 'Awards & Records', 'awards': 'Awards & Records',
    'press conference': 'Press Conference',
    'women': "Women's Football", "women's football": "Women's Football",
    'wsl': "Women's Football",
    # Basketball
    'nba': 'NBA', 'nba finals': 'NBA Finals', 'finals': 'NBA Finals',
    'playoffs': 'NBA Playoffs', 'nba playoffs': 'NBA Playoffs', 'play-offs': 'NBA Playoffs',
    'ncaa': 'NCAA', 'college basketball': 'NCAA', 'march madness': 'NCAA',
    'wnba': 'WNBA',
    'nba draft': 'NBA Draft', 'draft': 'NBA Draft',
    'nba trade': 'NBA Trades', 'trade': 'NBA Trades',
    # Tennis
    'wimbledon': 'Wimbledon',
    'us open': 'US Open', 'us open tennis': 'US Open', 'uso': 'US Open',
    'french open': 'French Open', 'roland garros': 'French Open', 'rg': 'French Open',
    'australian open': 'Australian Open', 'ao': 'Australian Open',
    'atp': 'ATP Tour', 'atp tour': 'ATP Tour', 'atp 500': 'ATP Tour', 'atp 250': 'ATP Tour',
    'wta': 'WTA Tour', 'wta tour': 'WTA Tour',
    'grand slam': 'Grand Slams', 'grand slams': 'Grand Slams',
    'davis cup': 'Davis Cup', 'billie jean king cup': 'Billie Jean King Cup',
    # F1
    'race': 'Race Report', 'race report': 'Race Report', 'grand prix': 'Race Report', 'gp': 'Race Report',
    'qualifying': 'Qualifying', 'quali': 'Qualifying', 'qualification': 'Qualifying',
    'driver': 'Driver News', 'driver news': 'Driver News',
    'constructor': 'Constructor Championship', 'constructors': 'Constructor Championship',
    'technical': 'Technical', 'car development': 'Technical', 'upgrade': 'Technical',
    'sprint': 'Sprint Race', 'sprint race': 'Sprint Race',
    'penalty': 'Penalties & Stewards', 'stewards': 'Penalties & Stewards',
    # Generic
    'general': 'General', 'other': 'General', 'news': 'General',
    'football': 'General',  # bare "football" with no further context
    'tennis': 'General',
    'basketball': 'General',
    'f1': 'General',
}

# Additional numeric / slug patterns to strip before lookup
_DIGIT_ONLY_RE = re.compile(r'^\d+$')
_SLUG_CLEAN_RE  = re.compile(r'[-_]+')


def normalise_subtopic(raw, sport=''):
    """
    Map a raw sub_topic string to a canonical label.
    Handles: exact matches, substring matches, numeric IDs,
    URL slugs, and empty/null values.
    """
    if not isinstance(raw, str) or not raw.strip():
        return 'General'

    # Strip numeric-only values (database IDs leaking into the field)
    cleaned = raw.strip()
    if _DIGIT_ONLY_RE.match(cleaned):
        return 'General'

    # Normalise slug separators → spaces, lower-case
    key = _SLUG_CLEAN_RE.sub(' ', cleaned).strip().lower()

    # Remove any remaining standalone digits from slug tokens
    # e.g. "football 3" → "football", "article 1234" → "article"
    key_no_digits = ' '.join(t for t in key.split() if not _DIGIT_ONLY_RE.match(t)).strip()
    if not key_no_digits:
        return 'General'

    # Exact match (prefer digit-stripped version first, then raw)
    if key_no_digits in SUBTOPIC_MAP:
        return SUBTOPIC_MAP[key_no_digits]
    if key in SUBTOPIC_MAP:
        return SUBTOPIC_MAP[key]

    # Substring / contained match
    for k, v in SUBTOPIC_MAP.items():
        if k in key_no_digits or k in key:
            return v

    # Last resort: title-case the cleaned string if it looks meaningful
    # (at least one alpha character, not just symbols/numbers)
    title = key_no_digits.strip().title()
    if title and any(c.isalpha() for c in title):
        return title

    return 'General'

def classify_subtopic(sport, headline, body_snippet):
    """Keyword rule engine — matches notebook exactly. First match wins."""
    rules = _SUBTOPIC_RULES.get(sport, [])
    if not rules:
        return sport
    text = (str(headline) + ' ' + str(body_snippet)).lower()
    for label, keywords in rules:
        if not keywords:
            return label
        for kw in keywords:
            if kw in text:
                return label
_SUBTOPIC_RULES = {
    'Football': [
        ('Premier League',   ['premier league','epl','man united','man city','liverpool',
                               'arsenal','chelsea','tottenham','spurs','newcastle',
                               'aston villa','everton','west ham','brighton','fulham',
                               'brentford','nottingham forest','bournemouth','wolves',
                               'ipswich','leicester','southampton','crystal palace']),
        ('Champions League', ['champions league','ucl','europa league','conference league',
                               'group stage','knockout','semi-final','final','real madrid',
                               'barcelona','bayern','psg','juventus','inter milan',
                               'ac milan','atletico','dortmund','porto','benfica']),
        ('La Liga',          ['la liga','laliga','real madrid','barcelona','atletico madrid',
                               'sevilla','valencia','real sociedad','villarreal','betis']),
        ('Bundesliga',       ['bundesliga','bayern munich','borussia dortmund','rb leipzig',
                               'bayer leverkusen','eintracht','werder bremen','freiburg','mainz']),
        ('Serie A',          ['serie a','juventus','inter milan','ac milan','napoli',
                               'roma','lazio','fiorentina','atalanta']),
        ('Ligue 1',          ['ligue 1','psg','paris saint-germain','marseille','lyon',
                               'monaco','lens','rennes','nice']),
        ('International',    ['world cup','euros','euro 2024','nations league','international',
                               'england','france','germany','brazil','argentina',
                               'spain national','qualifying','friendl']),
        ('MLS',              ['mls','major league soccer','inter miami','la galaxy','nycfc',
                               'seattle sounders','portland timbers','atlanta united']),
        ('Transfer News',    ['transfer','signing','sign','deal','fee','million','contract',
                               'bid','move','window','loan','permanent']),
        ('General Football', []),
    ],
    'Basketball': [
        ('NBA',               ['nba','lakers','celtics','warriors','bulls','heat','knicks',
                                'bucks','suns','nuggets','76ers','nets','raptors','clippers',
                                'spurs','mavericks','jazz','thunder','rockets','lebron',
                                'curry','durant','giannis','playoffs','nba finals','all-star']),
        ('NCAA',              ['ncaa','college basketball','march madness','final four',
                                'sweet sixteen','ncaab','duke','kentucky','kansas',
                                'north carolina','gonzaga','uconn','michigan state','unc',
                                'college','university']),
        ('WNBA',              ['wnba','las vegas aces','new york liberty','seattle storm',
                                'chicago sky','caitlin clark','women\'s basketball']),
        ('EuroLeague',        ['euroleague','eurocup','real madrid basketball','fenerbahce',
                                'cska','olympiacos','barca basket','anadolu efes']),
        ('General Basketball', []),
    ],
    'Tennis': [
        ('Grand Slams',       ['wimbledon','us open','french open','roland garros',
                                'australian open','grand slam','slam']),
        ('ATP Tour',          ['atp','masters 1000','indian wells','miami open','monte carlo',
                                'rome','madrid','canada','cincinnati','paris masters',
                                'world tour finals','djokovic','alcaraz','medvedev',
                                'zverev','tsitsipas','sinner','rublev','fritz']),
        ('WTA Tour',          ['wta','swiatek','sabalenka','gauff','rybakina','pegula',
                                'jabeur','keys','kvitova','women\'s','ladies']),
        ('Davis Cup / Fed Cup',['davis cup','billie jean king cup','bjk cup','fed cup',
                                 'team tennis','nation']),
        ('General Tennis',    []),
    ],
    'F1': [
        ('Race Report',       ['race','grand prix','gp','podium','win','winner','victory',
                                'finished','result','lap','fastest lap','dnf','collision']),
        ('Qualifying',        ['qualifying','quali','pole position','pole','grid',
                                'front row','q1','q2','q3']),
        ('Constructor Battle',['constructor','team championship','mclaren','ferrari',
                                'red bull','mercedes','aston martin','alpine','haas',
                                'williams','sauber','driver standings','championship battle']),
        ('Driver News',       ['verstappen','hamilton','leclerc','norris','sainz','russell',
                                'alonso','piastri','driver','seat','contract',
                                'retirement','move','deal']),
        ('Technical',         ['upgrade','floor','diffuser','front wing','rear wing',
                                'engine','power unit','pu','aerodynamic','technical',
                                'regulation','protest','appeal','penalty']),
        ('General F1',        []),
    ],
}
            

def load_and_prepare(filepath):
    """
    Full data loading, cleaning, and feature engineering pipeline.
    - Headline repeated 2x in full_text to boost headline TF-IDF weight.
    - has_body flag added for quality tracking.
    - Numeric tokens preserved (scores, years, fees are meaningful in sports).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    df['body']     = df['body'].fillna('').astype(str)
    df['headline'] = df['headline'].fillna('').astype(str)
    df['author']   = df.get('author',    pd.Series(['Unknown']*len(df))).fillna('Unknown').astype(str)
    df['date']     = df.get('date',      pd.Series(['']*len(df))).fillna('').astype(str)
    df['url']      = df.get('url',       pd.Series(['']*len(df))).fillna('').astype(str)
    df['sport']    = df['sport'].fillna('Unknown').astype(str)
    df['source']   = df['source'].fillna('Unknown').astype(str)

    df['sub_topic'] = df.apply(
        lambda r: classify_subtopic(
            r.get('sport', ''),
            r.get('headline', ''),
            r.get('body', '')[:300]
        ), axis=1
    )

    # FIX: headline weighted 2x so headline terms rank higher in TF-IDF
    df['full_text']           = df['headline'] + ' ' + df['headline'] + ' ' + df['body']
    df['body_word_count']     = df['body'].apply(lambda x: len(x.split()))
    df['headline_word_count'] = df['headline'].apply(lambda x: len(x.split()))
    # FIX: track body presence explicitly
    df['has_body'] = df['body'].str.strip().ne('') & df['body'].str.strip().ne('N/A')

    df = df.drop_duplicates(subset='headline', keep='first').reset_index(drop=True)
    return df


df = load_and_prepare(DATA_FILE)

SPORTS    = sorted(df['sport'].unique().tolist())
SOURCES   = sorted(df['source'].unique().tolist())
SUBTOPICS = sorted(df['sub_topic'].unique().tolist())


# =============================================================================
# 2.  IR PREPROCESSING  (no external tokeniser — pure regex)
# =============================================================================
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
from nltk.corpus import stopwords as _nltk_sw
from nltk.stem import WordNetLemmatizer

# Base: full NLTK English list (179 words — articles, pronouns,
# prepositions, conjunctions, auxiliaries, contractions)
STOPWORDS = set(_nltk_sw.words('english'))

# Sports-domain additions: generic sports journalism words that
# are so frequent they add no retrieval value
STOPWORDS |= {
    # Reporting verbs
    'said','says','say','told','tells','tell','reported','reports','report',
    'according','claimed','claims','claim','added','adds','add','announced',
    'confirmed','confirms','confirm','revealed','reveals','reveal',
    'stated','states','state','wrote','writes','write','described',
    # Modal / auxiliary overflow (NLTK misses some contractions)
    'would','could','should','might','must','shall','may',
    'also','yet','still','even','though','although','however','despite',
    'following','including','addition','furthermore','meanwhile','therefore',
    # Time words too generic for sports IR
    'week','weeks','day','days','month','months','year','years',
    'today','yesterday','tomorrow','recently','soon','already','later',
    'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
    'january','february','march','april','june','july','august',
    'september','october','november','december',
    # Generic sports journalism noise
    'game','games','match','matches','season','seasons',
    'player','players','team','teams','club','clubs','side','sides',
    'manager','managers','coach','coaches','squad','squads',
    'sport','sports','news','latest','update','updates',
    'result','results','score','scores',
    'new','old','good','great','big','best','top','high','low',
    'make','made','take','taken','get','got','go','going','went',
    'come','came','back','well','like','just','first','second','third',
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'article','read','click','watch','live','video','photo','gallery',
    # Scraper boilerplate that slips through
    'subscribe','newsletter','cookie','advertisement','follow','share',
}

_lemmatizer = WordNetLemmatizer()

def preprocess(text, do_stem=False):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return [_lemmatizer.lemmatize(t) for t in tokens]


def build_ir_engine(data):
    """
    Build all IR structures from a DataFrame.
    TF-IDF: 8,000-feature (1,2)-gram with sublinear TF.
    """
    data = data.copy()
    data['tokens'] = data['full_text'].apply(preprocess)

    inv_idx = defaultdict(set)
    for idx, row in data.iterrows():
        for tok in row['tokens']:
            inv_idx[tok].add(idx)

    tfidf_v = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, stop_words=list(STOPWORDS)
    )
    tfidf_m = tfidf_v.fit_transform(data['full_text'])

    doc_lengths = [len(t) for t in data['tokens']]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)

    return {
        'df': data, 'inv_idx': inv_idx,
        'tfidf_vec': tfidf_v, 'tfidf_mat': tfidf_m,
        'doc_lengths': doc_lengths, 'avg_dl': avg_dl,
    }


ir = build_ir_engine(df)
df = ir['df']


def cosine_search(query, top_n=10):
    qvec   = ir['tfidf_vec'].transform([query])
    scores = (ir['tfidf_mat'] @ qvec.T).toarray().flatten()
    ranked = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0][:top_n]


K1, B = 1.5, 0.75

def bm25_search(query, top_n=10):
    N_DOCS  = len(df)
    qtoks   = preprocess(query)
    candidates = set()
    for qt in qtoks:
        candidates |= ir['inv_idx'].get(qt, set())
    if not candidates:
        return []

    def _score(doc_idx):
        dl = ir['doc_lengths'][doc_idx]
        s  = 0.0
        for qt in qtoks:
            df_t = len(ir['inv_idx'].get(qt, set()))
            if df_t == 0:
                continue
            tf  = df['tokens'].iloc[doc_idx].count(qt)
            idf = math.log((N_DOCS - df_t + 0.5) / (df_t + 0.5) + 1)
            s  += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / ir['avg_dl']))
        return s

    scored = sorted([(i, _score(i)) for i in candidates], key=lambda x: -x[1])
    return [(int(i), round(s, 4)) for i, s in scored[:top_n] if s > 0]


def boolean_search(query):
    all_ids = set(df.index)
    for or_part in [p.strip() for p in query.upper().split(' OR ')]:
        current = None
        for term in [p.strip() for p in or_part.split(' AND ')]:
            if term.startswith('NOT '):
                word = term[4:].strip().lower()
                neg  = set(df.index) - ir['inv_idx'].get(word, set())
                current = neg if current is None else current & neg
            else:
                word = term.lower()
                ids  = ir['inv_idx'].get(word, set())
                current = ids if current is None else current & ids
        if current:
            all_ids &= current
    return sorted(all_ids)


def make_snippet(text, query_words, window=40):
    words = text.split()
    ql = [q.lower() for q in query_words if q]
    best_start, best_hits = 0, 0
    for i in range(len(words)):
        hits = sum(1 for w in words[i:i+window] if w.lower() in ql)
        if hits > best_hits:
            best_hits, best_start = hits, i
    snippet = ' '.join(words[best_start:best_start+window])
    for q in ql:
        snippet = re.sub(f'(?i)({re.escape(q)})', r'**\1**', snippet)
    return '...' + snippet + '...'


# =============================================================================
# 4.  AI CLASSIFIERS  — LinearSVC with proper held-out evaluation
# =============================================================================

def build_classifiers(data):
    """Train sport + sub-topic LinearSVC classifiers with 80/20 stratified split."""

    X       = data['full_text'].values
    y_sport = data['sport'].values

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X, y_sport, test_size=0.2, random_state=42, stratify=y_sport
    )

    sport_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, stop_words=list(STOPWORDS)
        )),
        ('clf', LinearSVC(max_iter=3000, C=1.0, dual=True))
    ])
    sport_pipe.fit(X_tr_s, y_tr_s)
    sport_test_preds = sport_pipe.predict(X_te_s)

    cv5      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sport_cv = cross_val_score(sport_pipe, X, y_sport, cv=cv5, scoring='accuracy')

    st_counts = data['sub_topic'].value_counts()
    valid_st  = st_counts[st_counts >= 3].index.tolist()
    data_st   = data[data['sub_topic'].isin(valid_st)].copy()

    X_st = data_st['full_text'].values
    y_st = data_st['sub_topic'].values

    X_tr_st, X_te_st, y_tr_st, y_te_st = train_test_split(
        X_st, y_st, test_size=0.2, random_state=42,
        stratify=y_st if len(set(y_st)) > 1 else None
    )

    subtopic_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, stop_words=list(STOPWORDS)
        )),
        ('clf', LinearSVC(max_iter=3000, C=2.0, dual=True))
    ])
    subtopic_pipe.fit(X_tr_st, y_tr_st)
    sub_test_preds = subtopic_pipe.predict(X_te_st)

    n_folds = min(3, len(set(y_st)))
    sub_cv  = cross_val_score(
        subtopic_pipe, X_st, y_st,
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    return {
        'sport_pipe':    sport_pipe,
        'subtopic_pipe': subtopic_pipe,
        'data_st':       data_st,
        'sport_cv':      sport_cv,
        'sub_cv':        sub_cv,
        'X_te_s':   X_te_s,  'y_te_s':  y_te_s,  'sport_test_preds':  sport_test_preds,
        'X_te_st':  X_te_st, 'y_te_st': y_te_st,  'sub_test_preds':    sub_test_preds,
    }


clfs          = build_classifiers(df)
sport_pipe    = clfs['sport_pipe']
subtopic_pipe = clfs['subtopic_pipe']
data_st       = clfs['data_st']


def get_top_features(text, pipe, n=10):
    tfidf   = pipe.named_steps['tfidf']
    clf     = pipe.named_steps['clf']
    label   = pipe.predict([text])[0]
    classes = list(clf.classes_)
    if label not in classes:
        return [], label
    ci   = classes.index(label)
    coef = clf.coef_[ci] if clf.coef_.shape[0] > 1 else clf.coef_[0]
    fns  = np.array(tfidf.get_feature_names_out())
    idxs = np.argsort(coef)[::-1][:n]
    return [(fns[i], round(float(coef[i]), 4)) for i in idxs if coef[i] > 0], label


def predict_with_confidence(text):
    s_scores_raw = sport_pipe.decision_function([text])[0]
    s_classes    = list(sport_pipe.classes_)
    s_exp        = np.exp(s_scores_raw - s_scores_raw.max())
    s_probs      = s_exp / s_exp.sum()
    sport_pred   = s_classes[np.argmax(s_probs)]
    sport_conf   = dict(zip(s_classes, (s_probs * 100).round(1)))

    st_scores_raw = subtopic_pipe.decision_function([text])[0]
    st_classes    = list(subtopic_pipe.classes_)
    st_exp        = np.exp(st_scores_raw - st_scores_raw.max())
    st_probs      = st_exp / st_exp.sum()
    sub_pred      = st_classes[np.argmax(st_probs)]
    top5_sub      = sorted(zip(st_classes, (st_probs*100).round(1)), key=lambda x: -x[1])[:5]

    sport_feats, _ = get_top_features(text, sport_pipe,    n=10)
    sub_feats,   _ = get_top_features(text, subtopic_pipe, n=10)

    return {
        'sport': sport_pred, 'sport_conf': sport_conf,
        'sub_topic': sub_pred, 'sub_top5': top5_sub,
        'sport_feats': sport_feats, 'sub_feats': sub_feats,
    }


# =============================================================================
# 5.  SCRAPER
# =============================================================================

# =============================================================================
# 5.  SCRAPER  (robots.txt compliant, matches notebook behaviour)
# =============================================================================
import urllib.request, urllib.error, urllib.parse, html as html_lib
from bs4 import BeautifulSoup as _BS

# ── robots.txt helpers ────────────────────────────────────────────────────────
_ROBOTS_CACHE = {}   # source → {'disallowed': set, 'crawl_delay': float}

def _fetch_robots(source, robots_url):
    if source in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[source]
    disallowed, crawl_delay, in_wildcard = set(), 2.0, False
    try:
        req = urllib.request.Request(robots_url,
              headers={'User-Agent': 'Mozilla/5.0 (compatible; SportsIRBot/1.0; academic project)'})
        with urllib.request.urlopen(req, timeout=10) as r:
            for raw_line in r.read().decode('utf-8', errors='ignore').splitlines():
                line = raw_line.split('#')[0].strip()
                if not line:
                    in_wildcard = False
                    continue
                lo = line.lower()
                if lo.startswith('user-agent:'):
                    in_wildcard = line.split(':', 1)[1].strip() == '*'
                elif lo.startswith('disallow:') and in_wildcard:
                    p = line.split(':', 1)[1].strip()
                    if p:
                        disallowed.add(p)
                elif lo.startswith('crawl-delay:') and in_wildcard:
                    try:
                        crawl_delay = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        print(f'[robots] Could not fetch {robots_url}: {e}')
    result = {'disallowed': disallowed, 'crawl_delay': crawl_delay}
    _ROBOTS_CACHE[source] = result
    return result

def _is_allowed(source, robots_url, url):
    rules = _fetch_robots(source, robots_url)
    path  = urllib.parse.urlparse(url).path
    return not any(path.startswith(p) for p in rules['disallowed'])

def _crawl_delay(source, robots_url):
    return _fetch_robots(source, robots_url).get('crawl_delay', 2.0)

# ── shared fetch ──────────────────────────────────────────────────────────────
_HEADERS = {
    'User-Agent':      'Mozilla/5.0 (compatible; SportsIRBot/1.0; academic project)',
    'Accept':          'text/html,application/xhtml+xml',
    'Accept-Language': 'en-GB,en;q=0.9',
}

def _fetch(url):
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as r:
            if r.status != 200:
                print(f'  [fetch] HTTP {r.status} — skipping: {url[:70]}')
                return ''
            return r.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f'  [fetch] Error fetching {url[:70]}: {e}')
        return ''

# ── author from JSON-LD (matches notebook's extract_author_from_jsonld) ───────
def _author_from_jsonld(soup):
    import json as _json
    for script in soup.find_all('script', {'type': 'application/ld+json'}):
        try:
            ld = _json.loads(script.string or '')
            if isinstance(ld, list): ld = ld[0]
            author = ld.get('author')
            if isinstance(author, dict):  return author.get('name')
            if isinstance(author, list) and author: return author[0].get('name')
        except Exception:
            continue
    return None

# ── date from JSON-LD ─────────────────────────────────────────────────────────
def _date_from_jsonld(soup):
    import json as _json
    for script in soup.find_all('script', {'type': 'application/ld+json'}):
        try:
            ld = _json.loads(script.string or '')
            if isinstance(ld, list): ld = ld[0]
            for key in ['datePublished', 'dateCreated', 'dateModified']:
                if key in ld:
                    return str(ld[key])[:10]
        except Exception:
            continue
    return None

# ── boilerplate filter (matches notebook's noise_phrases) ─────────────────────
_NOISE = [
    'cookie', 'privacy policy', 'terms of use', 'subscribe', 'sign up',
    'newsletter', 'javascript', 'please enable', 'advertisement',
    'stream sky sports with now', 'no contract, cancel anytime',
    'back pages podcast', 'play super 6', 'live scores, results and order of play',
    'powerful prediction model', 'sportsline', 'draftkings promo', 'kalshi promo',
    'dfs player', 'nascar betting picks', 'enter for free', 'super 6',
    'million in career', 'best bets at',
]



def _clean_body(paragraphs):
    out = []
    for p in paragraphs:
        p = p.strip()
        if len(p) < 30:
            continue
        if any(n in p.lower() for n in _NOISE):
            continue
        out.append(p)
    return ' '.join(out)

# ── BBC Sport scraper ─────────────────────────────────────────────────────────
MAX_ARTICLES_PER_PAIR = 25
MAX_PAGES_PER_PAIR    = 3

def _build_page_url(source, base_url, page_num):
    """Build listing page URL for page N — mirrors notebook's build_page_url."""
    if page_num == 1:
        return base_url
    if source == 'BBC Sport':
        return None  
    if source == 'Sky Sports':
        return base_url.rstrip('/') + f'/{page_num}'
    if source == 'CBS Sports':
        return base_url.rstrip('/') + f'/?page={page_num}'
    return None


def _collect_links(source, sport, base_listing_url, robots_url):
    """
    Multi-page link collector — mirrors notebook's collect_links().
    Fetches up to MAX_PAGES_PER_PAIR listing pages, stops early on:
      - None page URL (no pagination support)
      - non-200 response
      - duplicate page fingerprint (JS-rendered site)
      - no new links found
    """
    import hashlib
    delay = _crawl_delay(source, robots_url)
    all_links, seen, first_fp = [], set(), None

    CBS_SPORT_PATH = {
        'Football': '/soccer/', 'Tennis': '/tennis/',
        'Basketball': '/nba/', 'F1': '/nascar/',
    }
    SKY_SPORT_KW = {
        'Football': 'football', 'Tennis': 'tennis',
        'Basketball': 'basketball', 'F1': 'f1',
    }

    for page_num in range(1, MAX_PAGES_PER_PAIR + 1):
        page_url = _build_page_url(source, base_listing_url, page_num)
        if page_url is None:
            break

        if not _is_allowed(source, robots_url, page_url):
            break

        time.sleep(delay)
        html = _fetch(page_url)
        if not html:
            break

        # Duplicate-page fingerprint (catches JS-rendered pagination)
        fp = __import__('hashlib').md5(html[:2000].encode()).hexdigest()
        if page_num == 1:
            first_fp = fp
        elif fp == first_fp:
            break

        soup = _BS(html, 'html.parser')
        new_links = []

        for a in soup.find_all('a', href=True):
            href = a['href']

            if source == 'BBC Sport':
                if '/sport/' in href and '/articles/' in href:
                    full = href if href.startswith('http') else 'https://www.bbc.com' + href
                    if full not in seen and _is_allowed(source, robots_url, full):
                        new_links.append(full); seen.add(full)

            elif source == 'Sky Sports':
                full = href if 'skysports.com' in href else (
                    'https://www.skysports.com' + href if href.startswith('/') else None)
                if not full: continue
                skip = ['/live-blog/','/topic/','/membership','/streaming/',
                        '/subscribe','/watch/','/video/','/gallery/']
                if any(p in full.lower() for p in skip): continue
                path = full.rstrip('/').split('?')[0]
                if path.endswith('/news'): continue
                kw = SKY_SPORT_KW.get(sport, '')
                if (kw in full.lower()
                        and any(seg.isdigit() for seg in full.split('/'))
                        and '#' not in full
                        and 'skysports.com' in full
                        and full not in seen
                        and _is_allowed(source, robots_url, full)):
                    new_links.append(full); seen.add(full)

            elif source == 'CBS Sports':
                sport_path = CBS_SPORT_PATH.get(sport, '')
                if not sport_path: continue
                full = href if href.startswith('http') else 'https://www.cbssports.com' + href
                if (sport_path in full
                        and '/news/' in full
                        and 'cbssports.com' in full
                        and not full.rstrip('/').endswith('/news')
                        and 'betting' not in full
                        and full not in seen
                        and _is_allowed(source, robots_url, full)):
                    new_links.append(full); seen.add(full)

        if not new_links:
            break
        all_links.extend(new_links)

    return all_links


def scrape_bbc_sport(sport_path, sport_label, source_label='BBC Sport',
                     max_articles=MAX_ARTICLES_PER_PAIR):
    ROBOTS_URL   = 'https://www.bbc.com/robots.txt'
    listing_url  = f'https://www.bbc.com/sport/{sport_path}'
    delay        = _crawl_delay(source_label, ROBOTS_URL)

    links = _collect_links(source_label, sport_label, listing_url, ROBOTS_URL)
    links = links[:max_articles]

    articles = []
    for url in links:
        if not _is_allowed(source_label, ROBOTS_URL, url):
            continue
        time.sleep(1.5)
        a_html = _fetch(url)
        if not a_html:
            continue
        soup = _BS(a_html, 'html.parser')

        h1 = soup.find('h1')
        headline = h1.text.strip() if h1 else ''
        if not headline or len(headline) < 10:
            continue

        author = 'BBC Sport'
        byline = soup.find('div', {'data-component': 'byline-block'})
        if byline: author = byline.text.strip()
        if author == 'BBC Sport':
            meta = soup.find('meta', {'name': 'author'})
            if meta and meta.get('content'): author = meta['content']
        if author == 'BBC Sport':
            ld = _author_from_jsonld(soup)
            if ld: author = ld

        # Date
        date = None
        for prop in ['article:published_time', 'og:article:published_time']:
            tag = soup.find('meta', {'property': prop})
            if tag and tag.get('content'): date = tag['content'][:10]; break
        if not date:
            for name in ['date', 'pubdate', 'DC.date']:
                tag = soup.find('meta', {'name': name})
                if tag and tag.get('content'): date = tag['content'][:10]; break
        if not date:
            tag = soup.find('time', {'datetime': True})
            if tag: date = tag['datetime'][:10]
        if not date: date = _date_from_jsonld(soup)
        if not date:
            span = soup.find('span', {'data-testid': 'timestamp'})
            if span: date = span.text.strip()
        date = date or 'Unknown'

        # Body
        text_blocks = soup.find_all('div', {'data-component': 'text-block'})
        paras = [p.text.strip() for block in text_blocks for p in block.find_all('p')]
        if not paras:
            article = soup.find('article')
            if article: paras = [p.text.strip() for p in article.find_all('p')]
        body = _clean_body(paras)
        if len(body) < 50: continue

        sub_topic = classify_subtopic(sport_label, headline, body[:300])

        articles.append({
            'source': source_label, 'sport': sport_label, 'sub_topic': sub_topic,
            'headline': headline, 'author': author, 'date': date, 'body': body,
            'url': url, 'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    return articles


def scrape_sky_sports(sport_path, sport_label, source_label='Sky Sports', max_articles=MAX_ARTICLES_PER_PAIR):
    ROBOTS_URL  = 'https://www.skysports.com/robots.txt'
    listing_url = f'https://www.skysports.com/{sport_path}'
    delay       = _crawl_delay('Sky Sports', ROBOTS_URL)

    links = _collect_links('Sky Sports', sport_label, listing_url, ROBOTS_URL)
    links = links[:max_articles]

    articles = []
    for url in links:
        if not _is_allowed(source_label, ROBOTS_URL, url):
            continue
        time.sleep(1.5)
        a_html = _fetch(url)
        if not a_html:
            continue
        soup = _BS(a_html, 'html.parser')

        h1 = soup.find('h1')
        headline = h1.text.strip() if h1 else ''
        if not headline or len(headline) < 10:
            continue

        # Author
        author = 'Sky Sports'
        meta = soup.find('meta', {'name': 'author'})
        if meta and meta.get('content'): author = meta['content']
        if author == 'Sky Sports':
            ld = _author_from_jsonld(soup)
            if ld: author = ld
        if author == 'Sky Sports':
            tag = (soup.find('span', class_=lambda c: c and 'author' in ' '.join(c).lower()) or
                   soup.find('p',    class_=lambda c: c and 'author' in ' '.join(c).lower()))
            if tag: author = tag.text.strip()

        # Date
        date = None
        for prop in ['article:published_time', 'og:article:published_time']:
            tag = soup.find('meta', {'property': prop})
            if tag and tag.get('content'): date = tag['content'][:10]; break
        if not date:
            tag = soup.find('time', {'datetime': True})
            if tag: date = tag['datetime'][:10]
        if not date: date = _date_from_jsonld(soup)
        date = date or 'Unknown'

        # Body
        body_tag = (
            soup.find('div', class_='sdc-article-body') or
            soup.find('div', class_=lambda c: c and 'sdc-article-body' in ' '.join(c)) or
            soup.find('div', class_=lambda c: c and 'article-body' in ' '.join(c)) or
            soup.find('div', attrs={'data-testid': lambda v: v and 'article-body' in str(v).lower()}) or
            soup.find('article')
        )
        if body_tag:
            paras = [p.text.strip() for p in body_tag.find_all('p')]
        else:
            paras = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 60]
        body = _clean_body(paras)
        if len(body) < 50: continue

        sub_topic = classify_subtopic(sport_label, headline, body[:300])

        articles.append({
            'source': source_label, 'sport': sport_label, 'sub_topic': sub_topic,
            'headline': headline, 'author': author, 'date': date, 'body': body,
            'url': url, 'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    return articles


def scrape_cbs_sports(sport_path, sport_label, source_label='CBS Sports', max_articles=MAX_ARTICLES_PER_PAIR):
    ROBOTS_URL  = 'https://www.cbssports.com/robots.txt'
    listing_url = f'https://www.cbssports.com/{sport_path}/news/'
    delay       = _crawl_delay('CBS Sports', ROBOTS_URL)

    links = _collect_links('CBS Sports', sport_label, listing_url, ROBOTS_URL)
    links = links[:max_articles]

    articles = []
    for url in links:
        if not _is_allowed(source_label, ROBOTS_URL, url):
            continue
        time.sleep(1.5)
        a_html = _fetch(url)
        if not a_html:
            continue
        soup = _BS(a_html, 'html.parser')

        h1 = soup.find('h1')
        headline = h1.text.strip() if h1 else ''
        if not headline or len(headline) < 10:
            continue

        # Author — class → meta → JSON-LD
        author = 'CBS Sports'
        tag = (soup.find('span', class_=lambda c: c and 'author' in ' '.join(c).lower()) or
               soup.find('a',    class_=lambda c: c and 'author' in ' '.join(c).lower()))
        if tag: author = tag.text.strip()
        if author == 'CBS Sports':
            meta = soup.find('meta', {'name': 'author'})
            if meta and meta.get('content'): author = meta['content']
        if author == 'CBS Sports':
            ld = _author_from_jsonld(soup)
            if ld: author = ld

        # Date — same 5-layer fallback
        date = None
        for prop in ['article:published_time', 'og:article:published_time']:
            tag = soup.find('meta', {'property': prop})
            if tag and tag.get('content'): date = tag['content'][:10]; break
        if not date:
            for name in ['date', 'pubdate', 'DC.date']:
                tag = soup.find('meta', {'name': name})
                if tag and tag.get('content'): date = tag['content'][:10]; break
        if not date:
            tag = soup.find('time', {'datetime': True})
            if tag: date = tag['datetime'][:10]
        if not date: date = _date_from_jsonld(soup)
        date = date or 'Unknown'

        # Body — Article-bodyContent → body class → article
        body_tag = soup.find('div', class_='Article-bodyContent')
        if body_tag:
            paras = [p.text.strip() for p in body_tag.find_all('p')]
        else:
            body_tag = soup.find('div', attrs={'class': lambda c: c and 'body' in ' '.join(c).lower()})
            if body_tag:
                paras = [p.text.strip() for p in body_tag.find_all('p')]
            else:
                article = soup.find('article')
                paras = [p.text.strip() for p in article.find_all('p')] if article else []
        body = _clean_body(paras)
        if len(body) < 50: continue

        sub_topic = classify_subtopic(sport_label, headline, body[:300])

        articles.append({
            'source': source_label, 'sport': sport_label, 'sub_topic': sub_topic,
            'headline': headline, 'author': author, 'date': date, 'body': body,
            'url': url, 'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    return articles


SCRAPE_TARGETS = [
    (scrape_bbc_sport,  ('football',   'Football',   'BBC Sport',   MAX_ARTICLES_PER_PAIR)),
    (scrape_bbc_sport,  ('tennis',     'Tennis',     'BBC Sport',   MAX_ARTICLES_PER_PAIR)),
    (scrape_bbc_sport,  ('basketball', 'Basketball', 'BBC Sport',   MAX_ARTICLES_PER_PAIR)),
    (scrape_bbc_sport,  ('formula1',   'F1',         'BBC Sport',   MAX_ARTICLES_PER_PAIR)),
    (scrape_sky_sports, ('football',   'Football',   'Sky Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_sky_sports, ('tennis',     'Tennis',     'Sky Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_sky_sports, ('basketball', 'Basketball', 'Sky Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_sky_sports, ('f1',         'F1',         'Sky Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_cbs_sports, ('soccer',     'Football',   'CBS Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_cbs_sports, ('tennis',     'Tennis',     'CBS Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_cbs_sports, ('nba',        'Basketball', 'CBS Sports',  MAX_ARTICLES_PER_PAIR)),
    (scrape_cbs_sports, ('nascar',     'F1',         'CBS Sports',  MAX_ARTICLES_PER_PAIR)),
]

_scrape_status = {
    'running':   False,
    'log':       [],
    'results':   [],
    'progress':  0,
    'total':     len(SCRAPE_TARGETS),
}


def run_rescrape(save_path):
    global df, ir, clfs, sport_pipe, subtopic_pipe, data_st
    global SPORTS, SOURCES, SUBTOPICS

    _scrape_status['running']   = True
    _scrape_status['log']       = []
    _scrape_status['results']   = []
    _scrape_status['progress']  = 0
    _scrape_status['done']      = 0
    _scrape_status['new_count'] = 0

    def log(msg, level='info'):
        ts  = datetime.now().strftime('%H:%M:%S')
        _scrape_status['log'].append({'time': ts, 'msg': msg, 'level': level})
        print(f'[{ts}] {msg}')

    log('═══ Scraping pipeline started ═══')
    new_articles = []

    for fn, fn_args in SCRAPE_TARGETS:
        source_name = fn_args[2] if len(fn_args) > 2 else ('BBC Sport' if 'bbc' in fn.__name__ else 'Sky Sports')
        sport_name  = fn_args[1]
        label       = f'{source_name} / {sport_name}'
        log(f'▶ Scraping {label}…')

        try:
            batch = fn(*fn_args)
            count = len(batch)
            new_articles.extend(batch)
            _scrape_status['results'].append({
                'source': source_name, 'sport': sport_name,
                'count': count, 'status': '✓ OK'
            })
            log(f'  └ {count} articles collected', 'success')
        except Exception as e:
            _scrape_status['results'].append({
                'source': source_name, 'sport': sport_name,
                'count': 0, 'status': f'✗ {str(e)[:40]}'
            })
            log(f'  └ Error: {e}', 'error')

        _scrape_status['done']     += 1
        _scrape_status['progress']  = int(_scrape_status['done'] / _scrape_status['total'] * 90)

    log(f'═══ Collected {len(new_articles)} new articles total ═══')

    if new_articles:
        log('Merging with existing dataset…')
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []

        all_arts = existing + new_articles
        seen_url = set()
        deduped  = []
        for a in all_arts:
            url_key = a.get('url', '').strip()
            if url_key and url_key not in seen_url:
                seen_url.add(url_key)
                deduped.append(a)

        _scrape_status['new_count'] = len(deduped) - len(existing)
        log(f'After deduplication: {len(deduped)} total ({_scrape_status["new_count"]:+d} net new)')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)
        log(f'Saved to {save_path}')

        log('Rebuilding IR index and retraining classifiers…')
        df         = load_and_prepare(save_path)
        ir_new     = build_ir_engine(df)
        df         = ir_new['df']
        ir.update(ir_new)
        clfs_new   = build_classifiers(df)
        clfs.update(clfs_new)
        sport_pipe    = clfs['sport_pipe']
        subtopic_pipe = clfs['subtopic_pipe']
        data_st       = clfs['data_st']
        SPORTS    = sorted(df['sport'].unique().tolist())
        SOURCES   = sorted(df['source'].unique().tolist())
        SUBTOPICS = sorted(df['sub_topic'].unique().tolist())
        log('✓ Index and models refreshed successfully', 'success')
    else:
        log('No new articles collected — keeping existing dataset.', 'warn')

    _scrape_status['progress'] = 100
    _scrape_status['running']  = False
    _scrape_status['last']     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log('═══ Pipeline complete ═══', 'success')


# =============================================================================
# 6.  WORD CLOUD HELPER
# =============================================================================

def fig_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def make_wordcloud_img(sport_filter=None):
    subset = df if sport_filter is None else df[df['sport'] == sport_filter]
    tokens = [tok for row in subset['tokens'] for tok in row]
    if not tokens:
        return None
    wc = WordCloud(
        width=700, height=300, background_color='#1a0033',
        colormap='Purples', max_words=120, collocations=False,
        prefer_horizontal=0.8,
    ).generate(' '.join(tokens))
    return fig_to_b64(wc.to_image())


# =============================================================================
# 7.  PLOTLY THEME
# =============================================================================
PURPLE  = '#9b59b6'
DPURPLE = '#6c3483'
LPURPLE = '#d7bde2'
BLACK   = '#0d0d0d'
DARK    = '#1a0033'
CARD_BG = '#120024'
TEXT    = '#e8daef'
GRID    = '#2d0050'

SPORT_COLORS = {
    'Football':   '#9b59b6',
    'Tennis':     '#c39bd3',
    'Basketball': '#6c3483',
    'F1':         '#e8daef',
}

def base_layout(title=''):
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=15)),
        paper_bgcolor=CARD_BG, plot_bgcolor=DARK,
        font=dict(color=TEXT, family='Segoe UI, sans-serif'),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=DPURPLE, font_color='white'),
    )


# =============================================================================
# 8.  EDA FIGURE BUILDERS
# =============================================================================

def insight_p(text):
    return html.P(text, style={
        'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic',
        'marginTop': '6px', 'borderLeft': f'3px solid {DPURPLE}',
        'paddingLeft': '8px',
    })

def fig_sport_bar():
    counts = df['sport'].value_counts().reset_index()
    counts.columns = ['sport', 'count']
    total = counts['count'].sum()
    counts['pct'] = (counts['count'] / total * 100).round(1)
    fig = px.bar(counts, x='sport', y='count', color='sport',
                 color_discrete_map=SPORT_COLORS,
                 text=counts.apply(lambda r: f"{r['count']} ({r['pct']}%)", axis=1))
    fig.update_traces(textposition='outside', marker_line_color=BLACK, marker_line_width=1)
    fig.update_layout(**base_layout('Articles per Sport'), showlegend=False)
    top = counts.iloc[0]
    return fig, (f"{top['sport']} dominates with {top['count']} articles ({top['pct']}%). "
                 f"This reflects BBC Sport's stronger football coverage vs other sports.")

def fig_source_pie():
    counts = df['source'].value_counts().reset_index()
    counts.columns = ['source', 'count']
    fig = px.pie(counts, names='source', values='count',
                 color_discrete_sequence=[PURPLE, DPURPLE, LPURPLE], hole=0.4)
    fig.update_layout(**base_layout('Articles per Source'))
    fig.update_traces(textfont_color='white')
    top = counts.iloc[0]
    return fig, (f"{top['source']} contributes most articles ({top['count']}). "
                 f"Multi-source balancing reduces single-outlet bias in IR rankings.")

def fig_subtopic_bar():
    counts = df['sub_topic'].value_counts().reset_index()
    counts.columns = ['sub_topic', 'count']
    fig = px.bar(counts, x='count', y='sub_topic', orientation='h',
                 color='count',
                 color_continuous_scale=[[0, '#2d0050'], [1, '#9b59b6']],
                 text='count')
    fig.update_traces(textposition='outside')
    layout = base_layout('Articles per Sub-Topic')
    layout.pop('yaxis', None)
    fig.update_layout(**layout, coloraxis_showscale=False,
                      yaxis=dict(autorange='reversed', gridcolor=GRID, color=TEXT))
    top = counts.iloc[0]
    return fig, (f"'{top['sub_topic']}' is the most covered sub-topic ({top['count']} articles). "
                 f"Sub-topics with <3 articles are excluded from classifier training.")

def fig_source_sport_heatmap():
    cross = pd.crosstab(df['sport'], df['source'])
    fig = go.Figure(go.Heatmap(
        z=cross.values.tolist(), x=cross.columns.tolist(), y=cross.index.tolist(),
        colorscale=[[0, '#0d0d0d'], [0.5, '#6c3483'], [1, '#e8daef']],
        text=cross.values.tolist(), texttemplate='%{text}', showscale=False,
    ))
    fig.update_layout(**base_layout('Source × Sport Coverage Heatmap'))
    col_totals = cross.sum(axis=0)
    dominant   = col_totals.idxmax()
    return fig, (f"{dominant} has the broadest sport coverage. "
                 f"Empty cells = source does not cover that sport — can bias retrieval.")

def fig_headline_length():
    fig = go.Figure()
    for src in SOURCES:
        vals = df[df['source'] == src]['headline_word_count'].tolist()
        fig.add_trace(go.Box(y=vals, name=src,
                              marker_color=SPORT_COLORS.get(src, PURPLE),
                              line_color=LPURPLE))
    fig.update_layout(**base_layout('Headline Word Count by Source'))
    medians = {s: df[df['source']==s]['headline_word_count'].median() for s in SOURCES}
    longest = max(medians, key=medians.get)
    return fig, (f"{longest} writes the longest headlines (median {medians[longest]:.0f} words). "
                 f"Longer headlines provide richer TF-IDF features for that source.")

def fig_body_length_subtopic():
    order = (df[df['body_word_count'] > 0]
             .groupby('sub_topic')['body_word_count']
             .median().sort_values(ascending=False).index.tolist())
    fig = px.box(df[df['body_word_count'] > 50], x='sub_topic', y='body_word_count',
                 category_orders={'sub_topic': order},
                 color='sport', color_discrete_map=SPORT_COLORS)
    fig.update_layout(**base_layout('Body Word Count by Sub-Topic'))
    fig.update_xaxes(tickangle=-40)
    top_sub = order[0] if order else '—'
    return fig, (f"'{top_sub}' articles are longest on average — more indexable content "
                 f"per article. Short-body sub-topics may produce noisier TF-IDF vectors.")

def fig_top_keywords():
    all_tok = [tok for row in df['tokens'] for tok in row]
    freq = Counter(all_tok).most_common(20)
    words, counts = zip(*freq) if freq else ([], [])
    fig = px.bar(x=list(counts), y=list(words), orientation='h',
                 color=list(counts),
                 color_continuous_scale=[[0, '#2d0050'], [1, '#9b59b6']])
    layout = base_layout('Top 20 Keywords (All Articles)')
    layout.pop('yaxis', None)
    fig.update_layout(**layout, yaxis=dict(autorange='reversed'),
                      coloraxis_showscale=False)
    top3 = ', '.join(w for w, _ in freq[:3])
    return fig, (f"Top 3 terms: {top3}. High corpus frequency lowers IDF weight — "
                 f"rarer terms drive more precise retrieval in TF-IDF and BM25.")

def fig_keywords_per_sport():
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f'<b>{s}</b>' for s in SPORTS])
    positions = [(1,1),(1,2),(2,1),(2,2)]
    insights  = []
    for i, sport in enumerate(SPORTS):
        r, c  = positions[i]
        toks  = [tok for row in df[df['sport']==sport]['tokens'] for tok in row]
        freq  = Counter(toks).most_common(10)
        if not freq:
            continue
        w, cnt = zip(*freq)
        fig.add_trace(
            go.Bar(x=list(cnt)[::-1], y=list(w)[::-1], orientation='h',
                   marker_color=SPORT_COLORS.get(sport, PURPLE),
                   name=sport, showlegend=False),
            row=r, col=c
        )
        insights.append(f"{sport}: '{freq[0][0]}'")
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=DARK, font=dict(color=TEXT),
        title=dict(text='Top 10 Keywords per Sport', font=dict(color=TEXT, size=15)),
        height=480, margin=dict(l=40, r=20, t=60, b=30),
    )
    for ax in fig.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(gridcolor=GRID, zerolinecolor=GRID, color=TEXT)
    for ann in fig.layout.annotations:
        ann.font.color = LPURPLE
    note = ('Top keyword per sport — ' + ' | '.join(insights) +
            '. Sport-specific terms help the classifier separate categories.')
    return fig, note

def fig_has_body():
    counts = df['has_body'].value_counts().reset_index()
    counts.columns = ['has_body', 'count']
    counts['label'] = counts['has_body'].map({True: 'Has body text', False: 'Body missing'})
    fig = px.pie(counts, names='label', values='count',
                 color_discrete_sequence=[PURPLE, '#444'],  hole=0.4)
    fig.update_layout(**base_layout('Body Text Availability'))
    fig.update_traces(textfont_color='white')
    has  = df['has_body'].sum()
    miss = (~df['has_body']).sum()
    return fig, (f"{has} articles have body text; {miss} are headline-only. "
                 f"Missing body is caused by JS-rendered content on BBC Sport — "
                 f"static scraping cannot execute JavaScript.")


# =============================================================================
# 9.  EVALUATION FIGURES
# =============================================================================

def fig_confusion_matrix():
    labels = sorted(df['sport'].unique().tolist())
    cm     = confusion_matrix(clfs['y_te_s'], clfs['sport_test_preds'], labels=labels)
    fig    = go.Figure(go.Heatmap(
        z=cm.tolist(), x=labels, y=labels,
        colorscale=[[0, '#0d0d0d'], [0.5, '#6c3483'], [1, '#e8daef']],
        text=cm.tolist(), texttemplate='%{text}', showscale=False,
    ))
    fig.update_layout(**base_layout('Sport Classifier — Confusion Matrix (held-out test set)'))
    fig.update_yaxes(autorange='reversed')
    diag  = cm.diagonal().sum()
    total = cm.sum()
    return fig, (f"Evaluated on {total} held-out articles (20% of dataset). "
                 f"{diag}/{total} correctly classified. Off-diagonal = misclassifications, "
                 f"often cross-sport articles (e.g. NBA player transfers to football).")

def fig_cv_scores():
    sport_cv = clfs['sport_cv']
    sub_cv   = clfs['sub_cv']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(sport_cv))],
        y=sport_cv, name='Sport (5-fold CV)', marker_color=PURPLE,
        text=[f'{v:.2%}' for v in sport_cv], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(sub_cv))],
        y=sub_cv, name='Sub-Topic (CV)', marker_color=DPURPLE,
        text=[f'{v:.2%}' for v in sub_cv], textposition='outside',
    ))
    fig.add_hline(y=sport_cv.mean(), line_dash='dot', line_color=LPURPLE,
                  annotation_text=f'Sport avg {sport_cv.mean():.2%}')
    layout = base_layout('Cross-Validation Accuracy (generalisation estimate)')
    layout['yaxis'] = layout.get('yaxis', {})
    layout['yaxis'].update({'range': [0, 1.15], 'gridcolor': GRID, 'color': TEXT})
    fig.update_layout(**layout, barmode='group')
    return fig, (f"5-fold CV sport accuracy: {sport_cv.mean():.2%} ± {sport_cv.std():.2%}. "
                 f"Low fold variance = good generalisation, model not overfitting to one partition.")

def fig_subtopic_f1():
    labels = subtopic_pipe.classes_
    rep    = classification_report(
        clfs['y_te_st'], clfs['sub_test_preds'],
        labels=[l for l in labels if l in set(clfs['y_te_st'])],
        output_dict=True, zero_division=0
    )
    valid = [l for l in labels if l in rep]
    f1s   = [rep[l]['f1-score'] for l in valid]
    fig = px.bar(x=valid, y=f1s, color=f1s,
                 color_continuous_scale=[[0, '#2d0050'], [1, '#9b59b6']],
                 text=[f'{v:.2f}' for v in f1s])
    fig.update_traces(textposition='outside')
    layout = base_layout('Sub-Topic Classifier — F1 per Class (test set)')
    layout['xaxis'] = layout.get('xaxis', {})
    layout['xaxis'].update({'tickangle': -35, 'color': TEXT})
    fig.update_layout(**layout, coloraxis_showscale=False)
    low = [l for l, f in zip(valid, f1s) if f < 0.5]
    return fig, (f"Sub-topics with F1 < 0.5: {', '.join(low) if low else 'none'}. "
                 f"Low F1 = too few training examples or vocabulary overlap with other sub-topics.")

def fig_bm25_vs_tfidf():
    TEST_QUERIES = [
        ('Football',   'sport',     'Football'),
        ('Tennis',     'sport',     'Tennis'),
        ('Basketball', 'sport',     'Basketball'),
        ('F1',         'sport',     'F1'),
        ('Premier League', 'sub_topic', 'Premier League'),
        ('NBA',            'sub_topic', 'NBA'),
        ('Wimbledon',      'sub_topic', 'Wimbledon'),
        ('Race Report',    'sub_topic', 'Race Report'),
        ('transfer signing deal', 'sub_topic', 'Transfers & Rumours'),
        ('qualifying pole position grid', 'sub_topic', 'Qualifying'),
    ]
    rows = []
    for query, field, expected in TEST_QUERIES:
        tfidf_hits = cosine_search(query, top_n=5)
        bm25_hits  = bm25_search(query,  top_n=5)
        def p_at_5(hits):
            if not hits: return 0.0
            rel = sum(1 for idx, _ in hits
                      if expected.lower() in str(df.iloc[idx].get(field, '')).lower()
                      or expected.lower() in df.iloc[idx]['full_text'].lower())
            return rel / min(5, len(hits))
        rows.append({'Query': query, 'TF-IDF P@5': round(p_at_5(tfidf_hits), 2),
                     'BM25 P@5': round(p_at_5(bm25_hits), 2)})

    rdf = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='TF-IDF Cosine', x=rdf['Query'], y=rdf['TF-IDF P@5'],
                          marker_color=PURPLE, text=rdf['TF-IDF P@5'], textposition='outside'))
    fig.add_trace(go.Bar(name='BM25', x=rdf['Query'], y=rdf['BM25 P@5'],
                          marker_color=DPURPLE, text=rdf['BM25 P@5'], textposition='outside'))
    layout = base_layout('IR Evaluation — BM25 vs TF-IDF Cosine (Precision@5)')
    layout['xaxis'] = layout.get('xaxis', {})
    layout['xaxis'].update({'tickangle': -30, 'color': TEXT})
    layout['yaxis'] = layout.get('yaxis', {})
    layout['yaxis'].update({'range': [0, 1.3], 'gridcolor': GRID, 'color': TEXT})
    fig.update_layout(**layout, barmode='group')

    tfidf_avg = rdf['TF-IDF P@5'].mean()
    bm25_avg  = rdf['BM25 P@5'].mean()
    winner    = 'TF-IDF' if tfidf_avg >= bm25_avg else 'BM25'
    return fig, (f"Mean P@5 — TF-IDF: {tfidf_avg:.2f} | BM25: {bm25_avg:.2f}. "
                 f"{winner} performs better on average across {len(TEST_QUERIES)} test queries. "
                 f"BM25 handles term saturation better for long documents; "
                 f"TF-IDF cosine can edge ahead on short headlines.")


def fig_precision_at_k():
    results = []
    test_queries = (
        [(s, 'sport', s) for s in SPORTS] +
        [('Premier League', 'sub_topic', 'Premier League'),
         ('NBA', 'sub_topic', 'NBA'),
         ('Wimbledon', 'sub_topic', 'Wimbledon'),
         ('Race Report', 'sub_topic', 'Race Report')]
    )
    for query, field, expected in test_queries:
        hits    = cosine_search(query, top_n=5)
        if not hits:
            results.append({'Query': query, 'P@5': 0.0})
            continue
        relevant = sum(1 for idx, _ in hits
                       if expected.lower() in df.iloc[idx][field].lower()
                       or expected.lower() in df.iloc[idx]['full_text'].lower())
        results.append({'Query': query, 'P@5': relevant / min(5, len(hits))})
    rdf = pd.DataFrame(results)
    avg = rdf['P@5'].mean()
    fig = px.bar(rdf, x='Query', y='P@5', color='P@5',
                 color_continuous_scale=[[0, '#2d0050'], [1, '#9b59b6']],
                 text=[f'{v:.2f}' for v in rdf['P@5']], range_y=[0, 1.15])
    fig.update_traces(textposition='outside')
    fig.update_layout(**base_layout('IR Evaluation — Precision@5 per Query (TF-IDF)'),
                      coloraxis_showscale=False)
    return fig, (f"Mean P@5: {avg:.2f} across {len(test_queries)} queries. "
                 f"Scores above 0.8 indicate the index reliably surfaces relevant "
                 f"articles in the top-5 for sport-specific queries.")


# =============================================================================
# 10.  CLASSIFICATION REPORT TABLE
# =============================================================================

def build_clf_report_table(y_true, y_pred, labels):
    rep  = classification_report(y_true, y_pred, labels=labels,
                                  output_dict=True, zero_division=0)
    rows = []
    for lbl in labels:
        r = rep.get(lbl, {})
        rows.append({
            'Class':     lbl,
            'Precision': f"{r.get('precision', 0):.2f}",
            'Recall':    f"{r.get('recall', 0):.2f}",
            'F1-Score':  f"{r.get('f1-score', 0):.2f}",
            'Support':   int(r.get('support', 0)),
        })
    return rows

sport_report_rows = build_clf_report_table(
    clfs['y_te_s'], clfs['sport_test_preds'], sorted(SPORTS)
)
subtopic_report_rows = build_clf_report_table(
    clfs['y_te_st'], clfs['sub_test_preds'],
    sorted([l for l in subtopic_pipe.classes_ if l in set(clfs['y_te_st'])])
)


# =============================================================================
# 11.  LAYOUT HELPERS
# =============================================================================

TABLE_STYLE = {
    'style_table': {'overflowX': 'auto'},
    'style_cell': {
        'backgroundColor': CARD_BG, 'color': TEXT,
        'border': f'1px solid {GRID}', 'textAlign': 'left',
        'padding': '8px', 'fontFamily': 'Segoe UI, sans-serif',
        'fontSize': '13px', 'maxWidth': '300px',
        'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'normal',
    },
    'style_header': {
        'backgroundColor': DPURPLE, 'color': 'white',
        'fontWeight': 'bold', 'border': f'1px solid {DPURPLE}', 'textAlign': 'center',
    },
    'style_data_conditional': [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#1e0040'},
        {'if': {'state': 'selected'}, 'backgroundColor': PURPLE, 'border': f'1px solid {LPURPLE}'},
    ],
}

def card(children, style=None):
    base = {
        'background': CARD_BG, 'borderRadius': '10px',
        'padding': '16px', 'marginBottom': '16px',
        'border': f'1px solid {GRID}',
        'boxShadow': '0 4px 20px rgba(155,89,182,0.15)',
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def section_title(text):
    return html.H5(text, style={'color': LPURPLE, 'fontWeight': 'bold',
                                'marginBottom': '12px', 'letterSpacing': '0.5px'})

def make_chart_card(fig, insight, graph_id, cfg=None):
    return card([
        dcc.Graph(figure=fig, id=graph_id, config=cfg or {'displayModeBar': False}),
        html.P(insight, style={
            'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic',
            'marginTop': '4px', 'borderLeft': f'3px solid {DPURPLE}',
            'paddingLeft': '8px', 'marginBottom': '0',
        }),
    ])


# =============================================================================
# 12.  BUILD ALL FIGURES
# =============================================================================

def build_all_figures():
    f_sport,   i_sport   = fig_sport_bar()
    f_source,  i_source  = fig_source_pie()
    f_sub,     i_sub     = fig_subtopic_bar()
    f_heat,    i_heat    = fig_source_sport_heatmap()
    f_hl,      i_hl      = fig_headline_length()
    f_body,    i_body    = fig_body_length_subtopic()
    f_kw,      i_kw      = fig_top_keywords()
    f_kwsport, i_kwsport = fig_keywords_per_sport()
    f_hb,      i_hb      = fig_has_body()
    f_cm,      i_cm      = fig_confusion_matrix()
    f_cv,      i_cv      = fig_cv_scores()
    f_f1,      i_f1      = fig_subtopic_f1()
    f_p5,      i_p5      = fig_precision_at_k()
    f_cmp,     i_cmp     = fig_bm25_vs_tfidf()
    return {
        'sport':   (f_sport,   i_sport),   'source':  (f_source,  i_source),
        'sub':     (f_sub,     i_sub),     'heat':    (f_heat,    i_heat),
        'hl':      (f_hl,      i_hl),      'body':    (f_body,    i_body),
        'kw':      (f_kw,      i_kw),      'kwsport': (f_kwsport, i_kwsport),
        'hb':      (f_hb,      i_hb),
        'cm':      (f_cm,      i_cm),      'cv':      (f_cv,      i_cv),
        'f1':      (f_f1,      i_f1),      'p5':      (f_p5,      i_p5),
        'cmp':     (f_cmp,     i_cmp),
    }

FIGS = build_all_figures()


# =============================================================================
# 13.  SIDEBAR
# =============================================================================

sidebar = html.Div([
    html.Div([
        html.Div('⚽🎾🏀🏎️', style={'fontSize': '28px', 'textAlign': 'center', 'marginBottom': '4px'}),
        html.H4('Sports IR', style={'color': LPURPLE, 'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '2px'}),
        html.P('Phase 2 Dashboard', style={'color': '#888', 'textAlign': 'center', 'fontSize': '12px', 'marginBottom': '20px'}),
    ]),
    html.Hr(style={'borderColor': GRID}),
    dbc.Nav([
        dbc.NavLink([html.Span('📊 ', style={'marginRight': '6px'}), 'EDA Dashboard'],
                     href='#', id='nav-eda', active=True),
        dbc.NavLink([html.Span('🗃️ ', style={'marginRight': '6px'}), 'Data Explorer'],
                     href='#', id='nav-explorer'),
        dbc.NavLink([html.Span('🔍 ', style={'marginRight': '6px'}), 'IR Search'],
                     href='#', id='nav-ir'),
        dbc.NavLink([html.Span('🤖 ', style={'marginRight': '6px'}), 'AI Classifier'],
                     href='#', id='nav-ai'),
        dbc.NavLink([html.Span('📈 ', style={'marginRight': '6px'}), 'Evaluation'],
                     href='#', id='nav-eval'),
        dbc.NavLink([html.Span('🕷️ ', style={'marginRight': '6px'}), 'Live Scraper'],
                     href='#', id='nav-scraper'),
    ], vertical=True, pills=True),
    html.Hr(style={'borderColor': GRID, 'marginTop': '20px'}),
    html.Div(id='sidebar-stats', style={'padding': '8px'}),
], style={
    'width': '220px', 'minHeight': '100vh', 'background': '#0a001a',
    'padding': '20px 12px', 'position': 'fixed', 'left': 0, 'top': 0, 'bottom': 0,
    'borderRight': f'1px solid {GRID}', 'overflowY': 'auto',
})


# =============================================================================
# 14.  TAB LAYOUTS
# =============================================================================

# ── EDA Tab ──────────────────────────────────────────────────────────────────
tab_eda = html.Div([
    html.H4('📊 Exploratory Data Analysis', style={'color': LPURPLE, 'marginBottom': '20px'}),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['sport'],  'g-sport'),  md=6),
        dbc.Col(make_chart_card(*FIGS['source'], 'g-source'), md=6),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['hb'], 'g-hb'), md=4),
        dbc.Col(make_chart_card(*FIGS['hl'], 'g-hl'), md=8),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['sub'], 'g-sub'), md=12),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['heat'], 'g-heat'), md=6),
        dbc.Col(make_chart_card(*FIGS['body'], 'g-body'), md=6),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['kw'],      'g-kw'),      md=6),
        dbc.Col(make_chart_card(*FIGS['kwsport'], 'g-kwsport'), md=6),
    ]),
    card([
        section_title('🌥️ Word Cloud'),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='wc-sport-filter',
                options=[{'label': 'All Sports', 'value': 'ALL'}] +
                        [{'label': s, 'value': s} for s in SPORTS],
                value='ALL', clearable=False,
                style={'backgroundColor': DARK, 'color': BLACK},
            ), md=3),
        ], style={'marginBottom': '12px'}),
        html.Img(id='wordcloud-img', style={'width': '100%', 'borderRadius': '8px'}),
        html.P('Larger words = higher term frequency after stopword removal. Filter by sport to see its vocabulary signature.',
               style={'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '8px'}),
    ]),
], id='tab-eda', style={'display': 'block'})


# ── Data Explorer Tab ─────────────────────────────────────────────────────────
tab_explorer = html.Div([
    html.H4('🗃️ Data Explorer', style={'color': LPURPLE, 'marginBottom': '20px'}),
    card([
        section_title('Filters'),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='exp-sport',
                                  options=[{'label': 'All Sports', 'value': 'ALL'}] +
                                          [{'label': s, 'value': s} for s in SPORTS],
                                  value='ALL', clearable=False,
                                  style={'backgroundColor': DARK, 'color': BLACK}), md=3),
            dbc.Col(dcc.Dropdown(id='exp-source',
                                  options=[{'label': 'All Sources', 'value': 'ALL'}] +
                                          [{'label': s, 'value': s} for s in SOURCES],
                                  value='ALL', clearable=False,
                                  style={'backgroundColor': DARK, 'color': BLACK}), md=3),
            dbc.Col(dcc.Dropdown(id='exp-subtopic',
                                  options=[{'label': 'All Sub-Topics', 'value': 'ALL'}] +
                                          [{'label': s, 'value': s} for s in SUBTOPICS],
                                  value='ALL', clearable=False,
                                  style={'backgroundColor': DARK, 'color': BLACK}), md=4),
            dbc.Col(html.Button('Reset', id='exp-reset', n_clicks=0,
                                 style={'backgroundColor': DPURPLE, 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px',
                                        'padding': '8px 18px', 'cursor': 'pointer'}), md=2),
        ]),
    ]),
    card([
        html.Div(id='exp-count', style={'color': LPURPLE, 'marginBottom': '8px', 'fontSize': '13px'}),
        dash_table.DataTable(
            id='explorer-table',
            columns=[
                {'name': '#',         'id': 'idx'},
                {'name': 'Sport',     'id': 'sport'},
                {'name': 'Sub-Topic', 'id': 'sub_topic'},
                {'name': 'Source',    'id': 'source'},
                {'name': 'Headline',  'id': 'headline'},
                {'name': 'Date',      'id': 'date'},
                {'name': 'Author',    'id': 'author'},
            ],
            page_size=15, row_selectable='single',
            sort_action='native', filter_action='native',
            **TABLE_STYLE,
        ),
    ]),
    card([
        section_title('Article Detail'),
        html.Div(id='article-detail',
                  style={'color': TEXT, 'lineHeight': '1.8',
                         'whiteSpace': 'pre-wrap', 'fontSize': '14px'}),
    ]),
], id='tab-explorer', style={'display': 'none'})


# ── IR Search Tab ─────────────────────────────────────────────────────────────
tab_ir = html.Div([
    html.H4('🔍 Information Retrieval Search Engine', style={'color': LPURPLE, 'marginBottom': '20px'}),
    card([
        section_title('Search'),
        dbc.Row([
            dbc.Col(dcc.Input(
                id='ir-query', type='text',
                placeholder='e.g.  Guardiola Premier League  |  NBA AND playoffs  |  tennis NOT Wimbledon',
                debounce=False,
                style={'width': '100%', 'backgroundColor': DARK, 'color': TEXT,
                       'border': f'1px solid {PURPLE}', 'borderRadius': '6px',
                       'padding': '10px', 'fontSize': '14px'},
            ), md=7),
            dbc.Col(dcc.Dropdown(
                id='ir-mode',
                options=[
                    {'label': '📐 TF-IDF Cosine', 'value': 'tfidf'},
                    {'label': '📊 BM25',           'value': 'bm25'},
                    {'label': '🔢 Boolean',        'value': 'boolean'},
                ],
                value='tfidf', clearable=False,
                style={'backgroundColor': DARK, 'color': BLACK},
            ), md=2),
            dbc.Col(dcc.Dropdown(
                id='ir-sport-filter',
                options=[{'label': 'All Sports', 'value': 'ALL'}] +
                        [{'label': s, 'value': s} for s in SPORTS],
                value='ALL', clearable=False, placeholder='Filter sport',
                style={'backgroundColor': DARK, 'color': BLACK},
            ), md=2),
            dbc.Col(html.Button('Search', id='ir-search-btn', n_clicks=0,
                                 style={'backgroundColor': PURPLE, 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px',
                                        'padding': '10px', 'cursor': 'pointer',
                                        'width': '100%', 'fontWeight': 'bold'}), md=1),
        ]),
        html.P('TF-IDF & BM25: ranked by relevance score | Boolean: AND / OR / NOT (uppercase)',
               style={'color': '#888', 'fontSize': '12px', 'marginTop': '8px', 'marginBottom': 0}),
    ]),
    card([
        section_title('Results'),
        html.Div(id='ir-results-info', style={'color': LPURPLE, 'fontSize': '13px', 'marginBottom': '8px'}),
        html.Div(id='ir-results'),
    ]),
], id='tab-ir', style={'display': 'none'})


# ── AI Classifier Tab ─────────────────────────────────────────────────────────
tab_ai = html.Div([
    html.H4('🤖 Multi-Label AI Classifier', style={'color': LPURPLE, 'marginBottom': '4px'}),
    html.P(
        'Predicts Sport and Sub-Topic from any text using TF-IDF + LinearSVC. '
        'Fully offline — no external APIs. '
        'Confidence scores use softmax-normalised decision margins '
        '(LinearSVC does not produce true probabilities).',
        style={'color': '#aaa', 'fontSize': '13px', 'marginBottom': '20px'}
    ),
    card([
        section_title('Input Text'),
        dcc.Textarea(
            id='ai-input',
            placeholder='Paste or type a sports news headline or article body here…',
            style={'width': '100%', 'height': '140px', 'backgroundColor': DARK,
                   'color': TEXT, 'border': f'1px solid {PURPLE}', 'borderRadius': '6px',
                   'padding': '10px', 'fontSize': '14px', 'resize': 'vertical'},
        ),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Button('🔮 Classify', id='ai-classify-btn', n_clicks=0,
                                 style={'backgroundColor': PURPLE, 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px',
                                        'padding': '10px 28px', 'cursor': 'pointer',
                                        'fontWeight': 'bold', 'fontSize': '14px',
                                        'marginTop': '10px'}), md='auto'),
            dbc.Col(html.Button('🎲 Load Random Article', id='ai-random-btn', n_clicks=0,
                                 style={'backgroundColor': DPURPLE, 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px',
                                        'padding': '10px 20px', 'cursor': 'pointer',
                                        'fontSize': '13px', 'marginTop': '10px'}), md='auto'),
        ]),
    ]),
    html.Div(id='ai-results'),
    card([
        section_title('Model Selection Rationale'),
        dbc.Row([
            dbc.Col([
                html.P('Why LinearSVC?', style={'color': LPURPLE, 'fontWeight': 'bold', 'fontSize': '13px'}),
                html.Ul([
                    html.Li('Optimises hard-margin objective directly on sparse TF-IDF vectors — efficient and effective for short documents (headlines + bodies).',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.Li('Outperforms Naive Bayes (NB assumes feature independence — poor on n-grams) and matches Logistic Regression at far lower cost on high-dim sparse matrices.',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.Li('C=1.0 for sport (balanced 4-class); C=2.0 for sub-topic (more classes, tighter margin needed).',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                ], style={'paddingLeft': '16px'}),
            ], md=6),
            dbc.Col([
                html.P('Dimensionality note', style={'color': LPURPLE, 'fontWeight': 'bold', 'fontSize': '13px'}),
                html.Ul([
                    html.Li(f'8,000 features × ~{len(df)} articles — high feature-to-sample ratio is typical for text classification.',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.Li('min_df=2 filters hapax legomena, reducing effective dimensionality and mitigating overfitting risk.',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.Li('Cross-validation variance confirms the model generalises well despite high dimensionality.',
                            style={'color': TEXT, 'fontSize': '12px', 'marginBottom': '5px'}),
                ], style={'paddingLeft': '16px'}),
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.P('Sport Classifier', style={'color': LPURPLE, 'fontWeight': 'bold'}),
                html.P(f"5-Fold CV: {clfs['sport_cv'].mean():.2%} ± {clfs['sport_cv'].std():.2%}",
                       style={'color': TEXT, 'fontSize': '13px'}),
            ], md=6),
            dbc.Col([
                html.P('Sub-Topic Classifier', style={'color': LPURPLE, 'fontWeight': 'bold'}),
                html.P(f"Classes: {len(subtopic_pipe.classes_)} sub-topics (≥3 samples each)",
                       style={'color': TEXT, 'fontSize': '13px'}),
                html.P(f"CV: {clfs['sub_cv'].mean():.2%} ± {clfs['sub_cv'].std():.2%}",
                       style={'color': TEXT, 'fontSize': '13px'}),
            ], md=6),
        ]),
    ]),
], id='tab-ai', style={'display': 'none'})


# ── Evaluation Tab ────────────────────────────────────────────────────────────
tab_eval = html.Div([
    html.H4('📈 Model & IR Evaluation', style={'color': LPURPLE, 'marginBottom': '20px'}),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['cv'], 'g-cv'), md=6),
        dbc.Col(make_chart_card(*FIGS['cm'], 'g-cm'), md=6),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['f1'], 'g-f1'), md=12),
    ]),
    card([
        section_title('Sport Classifier — Classification Report (held-out test set)'),
        dash_table.DataTable(
            data=sport_report_rows,
            columns=[{'name': c, 'id': c} for c in ['Class','Precision','Recall','F1-Score','Support']],
            **TABLE_STYLE,
        ),
        html.P('Precision = of articles predicted as this sport, % correct. '
               'Recall = of actual articles for this sport, % found. '
               'F1 = harmonic mean. All on 20% held-out test set.',
               style={'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic',
                      'marginTop': '8px', 'borderLeft': f'3px solid {DPURPLE}', 'paddingLeft': '8px'}),
    ]),
    card([
        section_title('Sub-Topic Classifier — Classification Report (held-out test set)'),
        dash_table.DataTable(
            data=subtopic_report_rows,
            columns=[{'name': c, 'id': c} for c in ['Class','Precision','Recall','F1-Score','Support']],
            **TABLE_STYLE,
        ),
        html.P('Sub-topics with Support < 5 show lower F1 due to insufficient training examples. '
               'Label normalisation reduces class count and improves per-class support.',
               style={'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic',
                      'marginTop': '8px', 'borderLeft': f'3px solid {DPURPLE}', 'paddingLeft': '8px'}),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['cmp'], 'g-cmp'), md=12),
    ]),
    dbc.Row([
        dbc.Col(make_chart_card(*FIGS['p5'], 'g-p5'), md=12),
    ]),
    card([
        section_title('Limitations & Honest Discussion'),
        html.Ul([
            html.Li([html.Strong('No train/test leakage: '),
                     'Confusion matrix and reports use a proper 20% stratified held-out split, not training data.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
            html.Li([html.Strong('SVC confidence: '),
                     'LinearSVC does not output calibrated probabilities. Decision margins are softmax-normalised '
                     'as a confidence proxy — useful for ranking but not as true probabilities.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
            html.Li([html.Strong('Sub-topic sparsity: '),
                     'Sub-topics with ≤3 articles excluded from sub-topic classifier. '
                     'Re-scraping adds more data and may improve class coverage.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
            html.Li([html.Strong('English only: '),
                     'All sources are English. Queries in other languages will produce poor results.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
            html.Li([html.Strong('JS-rendered dates: '),
                     'All three sources render publication dates via JavaScript. '
                     'Static urllib scraping cannot execute JS — scraped_at is used as proxy. '
                     'Sky Sports URL date extraction (regex on path) partially mitigates this.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
            html.Li([html.Strong('High feature-to-sample ratio: '),
                     f'8,000 TF-IDF features with ~{len(df)} articles. '
                     'min_df=2 and cross-validation confirm this is manageable for LinearSVC '
                     'but PCA/SVD (LSI) could further improve generalisation on larger datasets.'],
                    style={'color': TEXT, 'marginBottom': '8px'}),
        ]),
    ]),
    card([
        section_title('IR Techniques Applied'),
        html.Ul([
            html.Li('Tokenisation — regex word splitting, no external tokeniser',          style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Stopword removal — 150-word custom list including sports noise terms', style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Lemmatisation — NLTK WordNetLemmatizer (matches notebook preprocessing)', style={'color': TEXT, 'marginBottom': '5px'}),            html.Li('Inverted index — term → document set, O(1) lookup per term',          style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('TF-IDF — 8,000-term (1,2)-gram, sublinear TF, min_df=2, headline 2× weighted', style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Cosine similarity — query vs document TF-IDF vector angle',            style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('BM25 — probabilistic model, k1=1.5, b=0.75, pure Python',             style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Boolean retrieval — AND / OR / NOT over inverted index',               style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Snippet generation — best-window extraction + query highlighting',     style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Multi-label text classification — TF-IDF + LinearSVC (OvR)',           style={'color': TEXT, 'marginBottom': '5px'}),
            html.Li('Sub-topic normalisation — canonical label mapping at ingest time',     style={'color': TEXT, 'marginBottom': '5px'}),
        ]),
    ]),
], id='tab-eval', style={'display': 'none'})


# ── Live Scraper Tab ──────────────────────────────────────────────────────────
tab_scraper = html.Div([
    html.H4('🕷️ Live Scraper', style={'color': LPURPLE, 'marginBottom': '4px'}),
    html.P(
        'Scrapes BBC Sport, Sky Sports, and CBS Sports — 3 sources × 4 sports = 12 targets. '
    'robots.txt is fetched and enforced per source. Up to 3 listing pages per target. '
    'Author, date, and body use multi-layer fallbacks matching the notebook pipeline exactly.',
        style={'color': '#aaa', 'fontSize': '13px', 'marginBottom': '20px'}
    ),
    card([
        section_title('Scrape Targets'),
        dash_table.DataTable(
            data=[{
                'Source': fn_args[2] if len(fn_args) > 2 else 'Sky Sports',
                'Sport':  fn_args[1],
                'Max Articles': fn_args[3] if len(fn_args) > 3 else fn_args[2] if len(fn_args) > 2 and isinstance(fn_args[2], int) else fn_args[-1],
                'Method': 'urllib + BeautifulSoup',
            } for fn, fn_args in SCRAPE_TARGETS],
            columns=[
                {'name': 'Source',       'id': 'Source'},
                {'name': 'Sport',        'id': 'Sport'},
                {'name': 'Max Articles', 'id': 'Max Articles'},
                {'name': 'Method',       'id': 'Method'},
            ],
            **TABLE_STYLE,
        ),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Button('🚀 Start Scraping', id='btn-rescrape', n_clicks=0,
                                 style={'backgroundColor': DPURPLE, 'color': 'white',
                                        'border': 'none', 'borderRadius': '8px',
                                        'padding': '12px 24px', 'cursor': 'pointer',
                                        'fontWeight': 'bold', 'fontSize': '14px',
                                        'width': '100%'}), md=3),
            dbc.Col(html.Div(id='scrape-status', style={'color': '#aaa', 'fontSize': '13px',
                                                          'padding': '12px 0'}), md=9),
        ]),
        dcc.Interval(id='scrape-poll', interval=2000, disabled=True),
    ]),
    html.Div(id='scraper-progress-section', children=[
        card([
            section_title('Progress'),
            html.Div(id='scrape-progress-bar-wrap', children=[
                html.Div(style={
                    'background': GRID, 'borderRadius': '6px', 'height': '18px',
                    'overflow': 'hidden', 'marginBottom': '10px',
                }, children=[
                    html.Div(id='scrape-progress-bar', style={
                        'background': f'linear-gradient(90deg, {DPURPLE}, {PURPLE})',
                        'height': '18px', 'width': '0%',
                        'transition': 'width 0.4s ease', 'borderRadius': '6px',
                    }),
                ]),
                html.Div(id='scrape-progress-label', style={'color': LPURPLE, 'fontSize': '13px'}),
            ]),
        ]),
    ], style={'display': 'none'}),
    html.Div(id='scraper-results-section', children=[
        card([
            section_title('Results per Target'),
            dash_table.DataTable(
                id='scrape-results-table',
                columns=[
                    {'name': 'Source',  'id': 'source'},
                    {'name': 'Sport',   'id': 'sport'},
                    {'name': 'Scraped', 'id': 'count'},
                    {'name': 'Status',  'id': 'status'},
                ],
                data=[],
                **TABLE_STYLE,
            ),
        ]),
    ], style={'display': 'none'}),
    html.Div(id='scraper-log-section', children=[
        card([
            section_title('Live Log'),
            html.Div(
                id='scrape-log-output',
                style={
                    'backgroundColor': '#050010',
                    'borderRadius': '6px',
                    'padding': '14px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px',
                    'height': '380px',
                    'overflowY': 'auto',
                    'border': f'1px solid {GRID}',
                },
            ),
        ]),
    ], style={'display': 'none'}),
    html.Div(id='scraper-summary-section', style={'display': 'none'}),
], id='tab-scraper', style={'display': 'none'})




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = 'Sports IR Dashboard'

app.layout = html.Div([
    sidebar,
    html.Div([
        dcc.Store(id='active-tab', data='eda'),
        tab_eda, tab_explorer, tab_ir, tab_ai, tab_eval, tab_scraper,
    ], style={
        'marginLeft': '240px', 'padding': '28px 24px',
        'minHeight': '100vh', 'backgroundColor': BLACK,
        'fontFamily': 'Segoe UI, sans-serif',
    }),
], style={'backgroundColor': BLACK, 'minHeight': '100vh'})


@app.callback(
    [Output('tab-eda',      'style'),
     Output('tab-explorer', 'style'),
     Output('tab-ir',       'style'),
     Output('tab-ai',       'style'),
     Output('tab-eval',     'style'),
     Output('tab-scraper',  'style'),
     Output('active-tab',   'data'),
     Output('nav-eda',      'active'),
     Output('nav-explorer', 'active'),
     Output('nav-ir',       'active'),
     Output('nav-ai',       'active'),
     Output('nav-eval',     'active'),
     Output('nav-scraper',  'active')],
    [Input('nav-eda',      'n_clicks'),
     Input('nav-explorer', 'n_clicks'),
     Input('nav-ir',       'n_clicks'),
     Input('nav-ai',       'n_clicks'),
     Input('nav-eval',     'n_clicks'),
     Input('nav-scraper',  'n_clicks')],
    prevent_initial_call=False,
)
def switch_tab(n1, n2, n3, n4, n5, n6):
    show, hide = {'display': 'block'}, {'display': 'none'}
    trigger = ctx.triggered_id or 'nav-eda'
    tabs = {
        'nav-eda':      [show, hide, hide, hide, hide, hide, 'eda',      True,  False, False, False, False, False],
        'nav-explorer': [hide, show, hide, hide, hide, hide, 'explorer', False, True,  False, False, False, False],
        'nav-ir':       [hide, hide, show, hide, hide, hide, 'ir',       False, False, True,  False, False, False],
        'nav-ai':       [hide, hide, hide, show, hide, hide, 'ai',       False, False, False, True,  False, False],
        'nav-eval':     [hide, hide, hide, hide, show, hide, 'eval',     False, False, False, False, True,  False],
        'nav-scraper':  [hide, hide, hide, hide, hide, show, 'scraper',  False, False, False, False, False, True],
    }
    return tabs.get(trigger, tabs['nav-eda'])


# ── Sidebar stats ─────────────────────────────────────────────────────────────
@app.callback(Output('sidebar-stats', 'children'),
              Input('scrape-poll', 'n_intervals'),
              Input('active-tab',  'data'))
def update_sidebar_stats(_, tab):
    return [
        html.P(f'📰 {len(df)} Articles',         style={'color': TEXT, 'margin': '4px 0', 'fontSize': '13px'}),
        html.P(f'🏅 {len(SPORTS)} Sports',        style={'color': TEXT, 'margin': '4px 0', 'fontSize': '13px'}),
        html.P(f'📡 {len(SOURCES)} Sources',       style={'color': TEXT, 'margin': '4px 0', 'fontSize': '13px'}),
        html.P(f'🏷️ {len(SUBTOPICS)} Sub-topics', style={'color': TEXT, 'margin': '4px 0', 'fontSize': '13px'}),
    ]


# ── Live Scraper callbacks ────────────────────────────────────────────────────
@app.callback(
    [Output('scrape-status',          'children'),
     Output('scrape-poll',            'disabled'),
     Output('btn-rescrape',           'disabled'),
     Output('scraper-progress-section','style'),
     Output('scraper-results-section', 'style'),
     Output('scraper-log-section',     'style'),
     Output('scrape-progress-bar',     'style'),
     Output('scrape-progress-label',   'children'),
     Output('scrape-results-table',    'data'),
     Output('scrape-log-output',       'children'),
     Output('scraper-summary-section', 'style'),
     Output('scraper-summary-section', 'children')],
    [Input('btn-rescrape', 'n_clicks'),
     Input('scrape-poll',  'n_intervals')],
    prevent_initial_call=False,
)
def handle_scraper(n_clicks, n_intervals):
    trigger    = ctx.triggered_id
    show_block = {'display': 'block'}
    hide       = {'display': 'none'}

    if trigger == 'btn-rescrape' and n_clicks and not _scrape_status['running']:
        t = threading.Thread(target=run_rescrape, args=(DATA_FILE,), daemon=True)
        t.start()

    prog = _scrape_status['progress']
    bar_style = {
        'background': f'linear-gradient(90deg, {DPURPLE}, {PURPLE})',
        'height': '18px', 'width': f'{prog}%',
        'transition': 'width 0.4s ease', 'borderRadius': '6px',
    }

    log_lines = []
    for entry in _scrape_status['log']:
        color_map = {'success': '#7dcea0', 'error': '#ec7063', 'warn': '#f0b27a', 'info': LPURPLE}
        color = color_map.get(entry['level'], LPURPLE)
        log_lines.append(
            html.Div([
                html.Span(f"[{entry['time']}] ", style={'color': '#666'}),
                html.Span(entry['msg'], style={'color': color}),
            ], style={'marginBottom': '3px'})
        )

    results_data = _scrape_status['results']
    done         = _scrape_status['done']
    total        = _scrape_status['total']
    progress_lbl = f"{done} / {total} targets scraped ({prog}%)"

    running = _scrape_status['running']
    has_log = len(_scrape_status['log']) > 0

    if running:
        status_text = f"⏳ Running… {done}/{total} targets"
        sections_visible = show_block
    elif has_log and not running:
        status_text = f"✓ Completed at {_scrape_status['last']}"
        sections_visible = show_block
    else:
        status_text = 'Click Start Scraping to fetch latest articles.'
        sections_visible = hide

    summary_style = show_block if (has_log and not running) else hide
    new_n = _scrape_status['new_count']
    summary = card([
        section_title('✅ Scrape Complete'),
        dbc.Row([
            dbc.Col([
                html.P('Net new articles', style={'color': '#888', 'fontSize': '12px', 'marginBottom': '2px'}),
                html.H3(f'+{new_n}', style={'color': LPURPLE, 'fontWeight': 'bold', 'margin': 0}),
            ], md=3),
            dbc.Col([
                html.P('Total in dataset', style={'color': '#888', 'fontSize': '12px', 'marginBottom': '2px'}),
                html.H3(str(len(df)), style={'color': LPURPLE, 'fontWeight': 'bold', 'margin': 0}),
            ], md=3),
            dbc.Col([
                html.P('Finished at', style={'color': '#888', 'fontSize': '12px', 'marginBottom': '2px'}),
                html.H3(_scrape_status['last'] or '—', style={'color': LPURPLE, 'fontWeight': 'bold',
                                                                'fontSize': '16px', 'margin': 0}),
            ], md=6),
        ]),
        html.P('IR index and classifiers have been retrained on the updated dataset.',
               style={'color': '#aaa', 'fontSize': '12px', 'marginTop': '10px', 'marginBottom': 0}),
    ]) if (has_log and not running) else html.Div()

    return (
        status_text,
        not running,
        running,
        sections_visible,
        sections_visible,
        sections_visible,
        bar_style,
        progress_lbl,
        results_data,
        log_lines,
        summary_style,
        summary,
    )


# ── Word cloud ────────────────────────────────────────────────────────────────
@app.callback(Output('wordcloud-img', 'src'),
              Input('wc-sport-filter', 'value'))
def update_wordcloud(sport):
    return make_wordcloud_img(None if sport == 'ALL' else sport) or ''


# ── Data Explorer ─────────────────────────────────────────────────────────────
@app.callback(
    [Output('explorer-table', 'data'),
     Output('exp-count',      'children')],
    [Input('exp-sport',    'value'),
     Input('exp-source',   'value'),
     Input('exp-subtopic', 'value'),
     Input('exp-reset',    'n_clicks')],
)
def update_explorer(sport, source, subtopic, reset):
    if ctx.triggered_id == 'exp-reset':
        sport = source = subtopic = 'ALL'
    filt = df.copy()
    if sport    != 'ALL': filt = filt[filt['sport']     == sport]
    if source   != 'ALL': filt = filt[filt['source']    == source]
    if subtopic != 'ALL': filt = filt[filt['sub_topic'] == subtopic]
    rows = []
    for i, (_, row) in enumerate(filt.iterrows()):
        rows.append({
            'idx': i+1, 'sport': row['sport'], 'sub_topic': row['sub_topic'],
            'source': row['source'], 'headline': row['headline'],
            'date': row['date'], 'author': row['author'],
            '_body': row['body'], '_url': row['url'],
        })
    return rows, f'Showing {len(rows)} of {len(df)} articles'


@app.callback(
    Output('article-detail', 'children'),
    Input('explorer-table',  'selected_rows'),
    State('explorer-table',  'data'),
)
def show_article(selected, data):
    if not selected or not data:
        return 'Click a row in the table above to read the full article.'
    row  = data[selected[0]]
    body = row.get('_body', '')
    url  = row.get('_url',  '')
    pred = predict_with_confidence(row['headline'] + ' ' + body[:500])
    ai_badge = html.Div([
        html.Span('🤖 AI: ', style={'color': '#888', 'fontSize': '12px'}),
        html.Span(pred['sport'],    style={'backgroundColor': DPURPLE, 'color': 'white',
                                           'borderRadius': '4px', 'padding': '2px 8px',
                                           'fontSize': '12px', 'marginRight': '6px'}),
        html.Span(pred['sub_topic'], style={'backgroundColor': GRID, 'color': LPURPLE,
                                            'borderRadius': '4px', 'padding': '2px 8px',
                                            'fontSize': '12px'}),
    ], style={'marginBottom': '10px'})
    return html.Div([
        html.H5(row['headline'], style={'color': LPURPLE, 'marginBottom': '8px'}),
        html.P(f"📰 {row['source']}  |  🏅 {row['sport']}  |  🏷️ {row['sub_topic']}  "
               f"|  📅 {row['date']}  |  ✍️ {row['author']}",
               style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '8px'}),
        ai_badge,
        html.P(body[:3000] + ('…' if len(body) > 3000 else ''),
               style={'color': TEXT, 'lineHeight': '1.8', 'fontSize': '14px'}),
        html.A('🔗 Open Original Article', href=url, target='_blank',
               style={'color': PURPLE, 'fontSize': '12px'}) if url else html.Span(),
    ])


# ── IR Search ─────────────────────────────────────────────────────────────────
@app.callback(
    [Output('ir-results',      'children'),
     Output('ir-results-info', 'children')],
    Input('ir-search-btn',  'n_clicks'),
    [State('ir-query',       'value'),
     State('ir-mode',        'value'),
     State('ir-sport-filter','value')],
    prevent_initial_call=True,
)
def run_search(n_clicks, query, mode, sport_filter):
    if not query or not query.strip():
        return html.P('Enter a query above and click Search.', style={'color': '#888'}), ''

    query       = query.strip()
    query_words = re.findall(r'[a-zA-Z]+', query.lower())

    if mode == 'tfidf':
        results, mode_label = cosine_search(query, top_n=15), 'TF-IDF Cosine Similarity'
    elif mode == 'bm25':
        results, mode_label = bm25_search(query, top_n=15), 'BM25'
    else:
        ids     = boolean_search(query)
        results = [(i, None) for i in ids[:20]]
        mode_label = 'Boolean (AND/OR/NOT)'

    if sport_filter and sport_filter != 'ALL':
        results = [(i, s) for i, s in results if df.iloc[i]['sport'] == sport_filter]

    if not results:
        return html.P('No results found.', style={'color': '#aaa'}), f'Mode: {mode_label} | 0 results'

    cards = []
    for rank, (idx, score) in enumerate(results, 1):
        row     = df.iloc[idx]
        snippet = make_snippet(row['full_text'], query_words, window=35)
        score_badge = f'Score: {score:.4f}' if score is not None else 'Boolean match'

        parts        = snippet.split('**')
        snippet_html = []
        for j, part in enumerate(parts):
            if j % 2 == 1:
                snippet_html.append(html.Mark(part, style={
                    'backgroundColor': DPURPLE, 'color': 'white',
                    'padding': '0 3px', 'borderRadius': '3px',
                }))
            else:
                snippet_html.append(part)

        pred    = predict_with_confidence(row['headline'])
        ai_span = html.Span([
            '  🤖 ',
            html.Span(pred['sport'],    style={'backgroundColor': DPURPLE, 'color': 'white',
                                                'borderRadius': '3px', 'padding': '1px 6px',
                                                'fontSize': '11px', 'marginRight': '4px'}),
            html.Span(pred['sub_topic'], style={'backgroundColor': GRID, 'color': LPURPLE,
                                                 'borderRadius': '3px', 'padding': '1px 6px',
                                                 'fontSize': '11px'}),
        ])

        cards.append(html.Div([
            dbc.Row([
                dbc.Col(html.Span(f'#{rank}', style={'color': PURPLE, 'fontWeight': 'bold',
                                                       'fontSize': '18px'}), md=1),
                dbc.Col([
                    html.P([
                        html.A(row['headline'],
                               href=row['url'] if row['url'] else '#',
                               target='_blank',
                               style={'color': TEXT, 'fontWeight': 'bold',
                                      'fontSize': '14px', 'textDecoration': 'none'}),
                    ], style={'marginBottom': '4px'}),
                    html.P(snippet_html, style={'color': '#ccc', 'fontSize': '13px', 'marginBottom': '6px'}),
                    html.P([
                        f"📰 {row['source']}  |  🏅 {row['sport']}  |  "
                        f"🏷️ {row['sub_topic']}  |  {score_badge}",
                        ai_span,
                    ], style={'color': '#888', 'fontSize': '12px', 'marginBottom': 0}),
                ], md=11),
            ]),
        ], style={
            'padding': '14px', 'marginBottom': '10px',
            'backgroundColor': DARK, 'borderRadius': '8px',
            'borderLeft': f'4px solid {PURPLE}',
        }))

    info = f'Mode: {mode_label}  |  {len(results)} result(s) for "{query}"'
    return cards, info


# ── AI Classifier ─────────────────────────────────────────────────────────────
@app.callback(
    [Output('ai-results', 'children'),
     Output('ai-input',   'value')],
    [Input('ai-classify-btn', 'n_clicks'),
     Input('ai-random-btn',   'n_clicks')],
    State('ai-input', 'value'),
    prevent_initial_call=True,
)
def run_classifier(n_classify, n_random, text):
    trigger = ctx.triggered_id

    if trigger == 'ai-random-btn':
        sample = df.sample(1).iloc[0]
        text   = sample['headline'] + '\n\n' + str(sample.get('body', ''))[:800]

    if not text or not text.strip():
        return card(html.P('Please enter some text first.', style={'color': '#aaa'})), text or ''

    pred = predict_with_confidence(text.strip())

    def conf_bar(cls, conf_pct, is_top, color):
        return html.Div([
            html.Span(cls, style={'color': TEXT, 'fontSize': '12px',
                                  'minWidth': '130px', 'display': 'inline-block'}),
            html.Div(
                html.Div(style={'width': f'{conf_pct:.1f}%', 'height': '12px',
                                'backgroundColor': color if is_top else GRID,
                                'borderRadius': '6px'}),
                style={'backgroundColor': DARK, 'borderRadius': '6px', 'flexGrow': 1,
                       'height': '12px', 'display': 'inline-block',
                       'verticalAlign': 'middle', 'width': '140px', 'marginLeft': '6px'},
            ),
            html.Span(f' {conf_pct:.1f}%', style={'color': '#aaa', 'fontSize': '11px', 'marginLeft': '6px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'})

    def feat_bars(feats, color):
        if not feats:
            return [html.P('—', style={'color': '#888'})]
        max_w = max(feats[0][1], 0.001)
        return [html.Div([
            html.Span(f, style={'color': TEXT, 'fontSize': '13px',
                                'minWidth': '130px', 'display': 'inline-block'}),
            html.Div(html.Div(style={
                'width': f'{min(w/max_w*100,100):.1f}%', 'height': '10px',
                'backgroundColor': color, 'borderRadius': '5px',
            }), style={'backgroundColor': DARK, 'borderRadius': '5px', 'display': 'inline-block',
                        'width': '120px', 'verticalAlign': 'middle', 'marginLeft': '8px'}),
            html.Span(f' {w:.4f}', style={'color': '#aaa', 'fontSize': '11px', 'marginLeft': '6px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'})
        for f, w in feats]

    result_card = card([
        section_title('🔮 Prediction Results'),
        dbc.Row([
            dbc.Col([
                html.P('🏅 Sport', style={'color': LPURPLE, 'fontSize': '15px', 'fontWeight': 'bold'}),
                html.H2(pred['sport'], style={'color': 'white', 'fontWeight': 'bold',
                                               'fontSize': '28px', 'marginBottom': '16px'}),
                html.P('Softmax-normalised confidence:', style={'color': '#aaa', 'fontSize': '12px'}),
                *[conf_bar(cls, conf, cls == pred['sport'], PURPLE)
                  for cls, conf in sorted(pred['sport_conf'].items(), key=lambda x: -x[1])],
            ], md=6),
            dbc.Col([
                html.P('🏷️ Sub-Topic', style={'color': LPURPLE, 'fontSize': '15px', 'fontWeight': 'bold'}),
                html.H2(pred['sub_topic'], style={'color': 'white', 'fontWeight': 'bold',
                                                   'fontSize': '22px', 'marginBottom': '16px'}),
                html.P('Top 5 candidates:', style={'color': '#aaa', 'fontSize': '12px'}),
                *[conf_bar(cls, conf, cls == pred['sub_topic'], DPURPLE)
                  for cls, conf in pred['sub_top5']],
            ], md=6),
        ]),
    ])

    feats_card = card([
        section_title('🔑 Top TF-IDF Features Driving Prediction'),
        dbc.Row([
            dbc.Col([
                html.P(f"Sport → {pred['sport']}", style={'color': LPURPLE, 'fontWeight': 'bold', 'fontSize': '13px'}),
                *feat_bars(pred['sport_feats'], PURPLE),
            ], md=6),
            dbc.Col([
                html.P(f"Sub-Topic → {pred['sub_topic']}", style={'color': LPURPLE, 'fontWeight': 'bold', 'fontSize': '13px'}),
                *feat_bars(pred['sub_feats'], DPURPLE),
            ], md=6),
        ]),
        html.P('LinearSVC coefficients for the predicted class — '
               'words with higher coefficients push the model more strongly toward that prediction.',
               style={'color': LPURPLE, 'fontSize': '12px', 'fontStyle': 'italic',
                      'marginTop': '8px', 'borderLeft': f'3px solid {DPURPLE}', 'paddingLeft': '8px'}),
    ])

    return html.Div([result_card, feats_card]), text


# =============================================================================
# 17.  RUN
# =============================================================================
if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Sports IR Dashboard — Phase 2')
    print(f'  Data : {DATA_FILE}')
    print(f'  Open : http://{args.host}:{args.port}')
    print('='*60 + '\n')
    app.run(debug=False, host=args.host, port=args.port)