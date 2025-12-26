# app.py
"""
Recipe classifier Flask server (ready-to-run)

Place the following next to this file:
 - data_cleaned2.json      (your dataset in JSON; see README in code comments)
 - pipeline.pkl            (optional, recommended: full pipeline)
 OR
 - tfidf_vectorizer.pkl
 - logreg_model.pkl

Install dependencies:
pip install flask flask-cors pandas nltk scikit-learn imbalanced-learn rapidfuzz

Run:
python app.py
"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# fuzzy matcher
from rapidfuzz import process, fuzz
import json, pandas as pd, re
from pathlib import Path
BASE_DIR = Path('.').resolve()
raw = json.loads((BASE_DIR/"data_cleaned2.json").read_text(encoding='utf-8'))

# Print first 5 raw entries (good to see structure)
if isinstance(raw, dict):
    items = list(raw.items())[:8]
    print("JSON is a dict — sample keys & values:")
    for k,v in items:
        print("KEY:", k, "VALUE (type):", type(v))
        print(v)
        print("---")
elif isinstance(raw, list):
    print("JSON is a list — sample items:")
    for i,v in enumerate(raw[:8]):
        print(i, type(v))
        print(v)
        print("---")
else:
    print("JSON structure is unexpected:", type(raw))

# Convert using the same normalization logic to see DataFrame head
def safe_clean_text(s):
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def normalize_row_dict(d):
    name_keys = ['name', 'title', 'dish', 'dish_name']
    ingr_keys = ['ingredients', 'ingredient', 'ing']
    cuisine_keys = ['cuisine', 'country', 'region']
    out={}
    for k in name_keys:
        if k in d and d[k]:
            out['name']=d[k]; break
    for k in ingr_keys:
        if k in d and d[k]:
            out['ingredients']=d[k]; break
    for k in cuisine_keys:
        if k in d and d[k]:
            out['cuisine']=d[k]; break
    out.setdefault('name', None); out.setdefault('ingredients', None); out.setdefault('cuisine', None)
    return out

rows=[]
if isinstance(raw, dict):
    for k,v in raw.items():
        if isinstance(v, dict):
            row=normalize_row_dict(v)
            if not row['name']: row['name']=k
            rows.append(row)
        else:
            rows.append({'name':str(k),'ingredients':None,'cuisine':None})
elif isinstance(raw, list):
    for item in raw:
        if isinstance(item, dict):
            rows.append(normalize_row_dict(item))
        else:
            rows.append({'name':str(item),'ingredients':None,'cuisine':None})
df=pd.DataFrame(rows)
print("DF columns:", df.columns.tolist())
print("DF head:")
print(df.head(20).to_dict(orient='records'))
print("Search keys sample:", [safe_clean_text(str(n)) for n in df['name'].astype(str).head(20).tolist()])


# Ensure NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Try to import external clean_text from jop if you have it
try:
    from jop import clean_text as external_clean_text
except Exception:
    external_clean_text = None

def fallback_clean_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def safe_clean_text(s: str) -> str:
    return external_clean_text(s) if external_clean_text else fallback_clean_text(s)

# -------- Configuration --------
BASE_DIR = Path(__file__).resolve().parent
DATA_JSON_PATH = BASE_DIR / "C:\\Users\\user\\Desktop\\pt\\data_cleaned2.json"   # your JSON dataset filename
PIPELINE_PATH = BASE_DIR / "C:\\Users\\user\\Desktop\\pt\\pipeline.pkl"         # optional pipeline including vectorizer+model
VECTORIZER_PATH = BASE_DIR / "C:\\Users\\user\\Desktop\\pt\\tfidf_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "C:\\Users\\user\\Desktop\\pt\\logreg_model.pkl"

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# -------- Text cleaning used for training (must match training) --------
lemmatizer = WordNetLemmatizer()
STOP_WORDS_CUSTOM = {'fresh', 'chopped', 'slices', 'diced', 'large', 'small', 'medium', 'optional'}

PHRASE_MAP = {
    'bell pepper': 'pepper',
    'capsicum': 'pepper',
    'tomato paste': 'tomato',
    'tomato sauce': 'tomato',
    'garlic clove': 'garlic',
    'olive oil': 'oil',
    'soy sauce': 'soy',
    'spring onion': 'onion',
}
_phrase_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in PHRASE_MAP.keys()) + r')\b', flags=re.IGNORECASE)
def apply_phrase_map(text: str) -> str:
    def _repl(m):
        return PHRASE_MAP[m.group(0).lower()]
    return _phrase_pattern.sub(_repl, text)

def clean_ingredients(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = apply_phrase_map(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if w not in STOP_WORDS_CUSTOM]
    return ' '.join(words)

# -------- Load model/pipeline --------
model = None
vectorizer = None
pipeline_loaded = False

if PIPELINE_PATH.exists():
    try:
        with open(PIPELINE_PATH, 'rb') as f:
            model = pickle.load(f)
            pipeline_loaded = True
            print(f"[MODEL] Loaded pipeline from {PIPELINE_PATH}")
    except Exception as e:
        print("[MODEL] Failed to load pipeline.pkl:", e)

if not pipeline_loaded:
    if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"[MODEL] Loaded vectorizer and model from {VECTORIZER_PATH} & {MODEL_PATH}")
        except Exception as e:
            print("[MODEL] Failed to load vectorizer/model:", e)
            model = None
    else:
        print("[MODEL] No pipeline or vectorizer+model found. Prediction endpoint will be unavailable until model files are added.")

# -------- Load JSON dataset and build search structures --------
df: Optional[pd.DataFrame] = None
SEARCH_KEYS = []
SEARCH_KEY_TO_IDX = {}

def normalize_row_dict(d: dict) -> dict:
    name_keys = ['name', 'title', 'dish', 'dish_name']
    ingr_keys = ['ingredients', 'ingredient', 'ing']
    cuisine_keys = ['cuisine', 'country', 'region']

    normalized = {}
    for k in name_keys:
        if k in d and d[k]:
            normalized['name'] = str(d[k])
            break
    for k in ingr_keys:
        if k in d and d[k]:
            normalized['ingredients'] = str(d[k])
            break
    for k in cuisine_keys:
        if k in d and d[k]:
            normalized['cuisine'] = str(d[k])
            break
    normalized.setdefault('name', None)
    normalized.setdefault('ingredients', None)
    normalized.setdefault('cuisine', None)
    return normalized

if DATA_JSON_PATH.exists():
    try:
        raw = json.loads(DATA_JSON_PATH.read_text(encoding='utf-8'))
        rows = []
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    row = normalize_row_dict(v)
                    if not row['name']:
                        row['name'] = str(k)
                    rows.append(row)
                else:
                    rows.append({'name': str(k), 'ingredients': None, 'cuisine': None})
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    rows.append(normalize_row_dict(item))
                else:
                    rows.append({'name': str(item), 'ingredients': None, 'cuisine': None})
        else:
            raise ValueError("Unrecognized JSON structure (expect list or dict).")
        df = pd.DataFrame(rows).dropna(subset=['name']).reset_index(drop=True)
        df['_search_key'] = df['name'].astype(str).apply(safe_clean_text)
        SEARCH_KEYS = df['_search_key'].tolist()
        SEARCH_KEY_TO_IDX = {}
        for idx, k in enumerate(SEARCH_KEYS):
            if k not in SEARCH_KEY_TO_IDX:
                SEARCH_KEY_TO_IDX[k] = idx
        print(f"[DATA] Loaded JSON dataset from {DATA_JSON_PATH} rows={len(df)}")
    except Exception as e:
        print("[DATA] Failed to load JSON dataset:", e)
        df = None
else:
    print(f"[DATA] JSON dataset not found at {DATA_JSON_PATH}. Lookup will be limited.")

# -------- Fuzzy helper --------
def fuzzy_suggestions(query_key: str, limit: int = 6, score_cutoff: int = 55):
    if not SEARCH_KEYS:
        return []
    results = process.extract(query_key, SEARCH_KEYS, scorer=fuzz.WRatio, limit=limit)
    out = []
    for match_key, score, _ in results:
        if score >= score_cutoff:
            df_idx = SEARCH_KEY_TO_IDX.get(match_key)
            out.append((match_key, int(score), int(df_idx) if df_idx is not None else None))
    return out

# -------- Flask app --------
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline_loaded,
        "dataset_rows": len(df) if df is not None else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST JSON { "ingredients": "..." }
    Returns { "prediction": "...", "confidence": 0.82 }
    """
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body."}), 400

    ingredients = payload.get('ingredients') or payload.get('text') or ""
    ingredients_cleaned = clean_ingredients(ingredients)

    try:
        if pipeline_loaded:
            # pipeline likely expects raw (string) input; we supply cleaned string
            X = [ingredients_cleaned]
            pred = model.predict(X)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(X)[0]
                idx = list(model.classes_).index(pred) if hasattr(model, "classes_") else None
                prob = float(prob_arr[idx]) if idx is not None else None
        else:
            X_vec = vectorizer.transform([ingredients_cleaned])
            pred = model.predict(X_vec)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(X_vec)[0]
                idx = list(model.classes_).index(pred) if hasattr(model, "classes_") else None
                prob = float(prob_arr[idx]) if idx is not None else None

        return jsonify({"prediction": str(pred), "confidence": prob})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route('/lookup', methods=['POST'])
def lookup():
    """
    POST JSON { "dish_name": "..." }
    Returns detailed matches and suggestions.
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body."}), 400

    name = payload.get('dish_name') or payload.get('name') or ""
    key = safe_clean_text(name)
    debug = {"query_raw": name, "query_key": key}

    if df is None:
        return jsonify({"found": False, "error": "Dataset not loaded on server.", "debug": debug}), 200

    # exact normalized match
        # exact normalized match
    exact = df[df['_search_key'] == key]
    if not exact.empty:
        r = exact.iloc[0]
        # Safest retrieval: use .get with defaults and keep raw row for debugging
        raw_row = r.to_dict()
        name_val = raw_row.get('name') or raw_row.get('title') or raw_row.get('dish') or None
        ingredients_val = raw_row.get('ingredients') if pd.notna(raw_row.get('ingredients')) else None
        cuisine_val = raw_row.get('cuisine') if pd.notna(raw_row.get('cuisine')) else None

        return jsonify({
            "found": True,
            "exact": {"name": name_val, "ingredients": ingredients_val, "cuisine": cuisine_val},
            "raw_row": raw_row,               # <-- helpful for debugging on client side
            "substring_matches": [],
            "suggestions": [],
            "debug": debug
        })


    # substring matches
    substr = df[df['_search_key'].str.contains(key, na=False)]
    substr_out = []
    if not substr.empty:
        for _, r in substr.head(8).iterrows():
            substr_out.append({"name": str(r['name']), "ingredients": str(r['ingredients']) if pd.notna(r['ingredients']) else None, "cuisine": str(r['cuisine']) if pd.notna(r['cuisine']) else None})


    for _, r in substr.head(8).iterrows():
        raw = r.to_dict()
        substr_out.append({
            "name": raw.get('name') or None,
            "ingredients": raw.get('ingredients') if pd.notna(raw.get('ingredients')) else None,
            "cuisine": raw.get('cuisine') if pd.notna(raw.get('cuisine')) else None,
            "raw_row": raw
        })
        r = df.iloc[idx]
        raw = r.to_dict()
        suggestions_out.append({
            "match_key": match_key,
            "score": score,
            "name": raw.get('name') or None,
            "ingredients": raw.get('ingredients') if pd.notna(raw.get('ingredients')) else None,
            "cuisine": raw.get('cuisine') if pd.notna(raw.get('cuisine')) else None,
            "raw_row": raw
        })

    # fuzzy suggestions
    suggestions = fuzzy_suggestions(key, limit=6, score_cutoff=55)
    suggestions_out = []
    for match_key, score, idx in suggestions:
        if idx is None or idx >= len(df):
            continue
        r = df.iloc[idx]
        suggestions_out.append({
            "match_key": match_key,
            "score": score,
            "name": str(r['name']),
            "ingredients": str(r['ingredients']) if pd.notna(r['ingredients']) else None,
            "cuisine": str(r['cuisine']) if pd.notna(r['cuisine']) else None
        })

    found = bool(substr_out or suggestions_out)
    return jsonify({
        "found": found,
        "exact": None,
        "substring_matches": substr_out,
        "suggestions": suggestions_out,
        "debug": debug
    }), 200

if __name__ == '__main__':
    print(f"Starting Flask server on http://{HOST}:{PORT} (debug={DEBUG})")
    app.run(host=HOST, port=PORT, debug=DEBUG)
