"""
=============================================================
  CyberShield — Full Training Script
  
  Trains 5 models from 2 datasets:

  Dataset 1: cyberbullying_tweets.csv + augmented_data.csv
    → model_xgboost.pkl      (main classifier)
    → model_svm.pkl          (main classifier)
    → model_lr.pkl           (main classifier)
    → label_encoder.pkl

  Dataset 2: measuring-hate-speech.parquet (Berkeley)
    → severity_model.pkl     (0-10 severity regressor)
    → sarcasm_model.pkl      (sarcasm/implicit binary classifier)

  Run: python train_model.py
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── NLTK ─────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── Config ───────────────────────────────────────────────────
CB_PATH       = "cyberbullying_tweets.csv"
AUG_PATH      = "augmented_data.csv"
BERKELEY_PATH = "measuring-hate-speech.parquet"
CB_TEXT_COL   = "tweet_text"
CB_LABEL_COL  = "cyberbullying_type"
SBERT_MODEL   = "all-mpnet-base-v2"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

CLASSES = [
    'age', 'ethnicity', 'gender',
    'not_cyberbullying', 'other_cyberbullying', 'religion',
]

# ── Preprocessing ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens).strip()

# ════════════════════════════════════════════════════════════
#  PART A — MAIN CLASSIFICATION MODELS
#  Dataset: cyberbullying_tweets.csv + augmented_data.csv
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART A — MAIN CLASSIFICATION PIPELINE")
print("="*60)

# ── A1. Load Twitter dataset ──────────────────────────────────
print("\n  [A1/5] Loading cyberbullying_tweets.csv...")
df = pd.read_csv(CB_PATH)
print(f"         Raw shape : {df.shape}")
df = df[[CB_TEXT_COL, CB_LABEL_COL]].dropna()
df.columns = ['text', 'label']
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.strip() != '']

# ── A2. Merge augmented data ──────────────────────────────────
print("\n  [A2/5] Merging augmented_data.csv...")
if os.path.exists(AUG_PATH):
    df_aug = pd.read_csv(AUG_PATH)
    df_aug.columns = ['text', 'label']
    df_aug['text'] = df_aug['text'].apply(clean_text)
    df_aug = df_aug[df_aug['text'].str.strip() != '']
    # Repeat 5x so formal English has meaningful gradient weight
    df_aug = pd.concat([df_aug] * 5, ignore_index=True)
    df = pd.concat([df, df_aug], ignore_index=True)
    print(f"         Augmentation added: {len(df_aug)} formal sentences")
else:
    print("         augmented_data.csv not found — skipping")

df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"         Final shape : {df.shape}")
print(f"         Distribution:\n{df['label'].value_counts()}\n")

# ── A3. Encode labels & split ─────────────────────────────────
print("  [A3/5] Encoding labels & splitting...")
le = LabelEncoder()
le.fit(CLASSES)
df['label_enc'] = le.transform(df['label'])
joblib.dump(le, 'label_encoder.pkl')
print(f"         Classes: {list(le.classes_)}")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'].to_numpy(),
    df['label_enc'].to_numpy(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['label_enc'].to_numpy()
)
print(f"         Train: {len(X_train_text):,} | Test: {len(X_test_text):,}")

# ── A4. SBERT encoding ────────────────────────────────────────
print(f"\n  [A4/5] SBERT encoding ({SBERT_MODEL})...")
print("         Model already downloaded — encoding only.")
sbert = SentenceTransformer(SBERT_MODEL)

print("         Encoding TRAIN set...")
t0 = time.time()
X_train_emb = sbert.encode(
    X_train_text.tolist(),
    show_progress_bar=True,
    batch_size=32
)
print(f"         Done in {(time.time()-t0)/60:.1f} mins")

print("         Encoding TEST set...")
X_test_emb = sbert.encode(
    X_test_text.tolist(),
    show_progress_bar=True,
    batch_size=32
)

joblib.dump(X_train_emb, 'sbert_embeddings_train.pkl')
joblib.dump(X_test_emb,  'sbert_embeddings_test.pkl')
joblib.dump(X_test_emb,  'X_test_embeddings.pkl')
joblib.dump(y_test,       'y_test.pkl')
print("         Embeddings saved.")

# ── SMOTE ─────────────────────────────────────────────────────
print("\n         Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_emb, y_train)
print(f"         Before: {len(X_train_emb):,} → After: {len(X_train_bal):,}")

# ── A5. Train classifiers ─────────────────────────────────────
print("\n  [A5/5] Training 3 classifiers...")
results_cls = {}

print("\n    → XGBoost...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=0
)
xgb.fit(X_train_bal, y_train_bal)
joblib.dump(xgb, 'model_xgboost.pkl')
preds = xgb.predict(X_test_emb)
results_cls['XGBoost'] = accuracy_score(y_test, preds)
print(f"       Accuracy: {results_cls['XGBoost']:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

print("\n    → SVM (C=5.0)...")
svm = SVC(kernel='rbf', probability=True, C=5.0, random_state=RANDOM_STATE)
svm.fit(X_train_bal, y_train_bal)
joblib.dump(svm, 'model_svm.pkl')
preds = svm.predict(X_test_emb)
results_cls['SVM'] = accuracy_score(y_test, preds)
print(f"       Accuracy: {results_cls['SVM']:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

print("\n    → Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
lr.fit(X_train_bal, y_train_bal)
joblib.dump(lr, 'model_lr.pkl')
preds = lr.predict(X_test_emb)
results_cls['Logistic Regression'] = accuracy_score(y_test, preds)
print(f"       Accuracy: {results_cls['Logistic Regression']:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

# ════════════════════════════════════════════════════════════
#  PART B — BERKELEY SEVERITY + SARCASM MODELS
#  Dataset: measuring-hate-speech.parquet
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART B — BERKELEY SEVERITY + SARCASM PIPELINE")
print("="*60)

# ── B1. Load Berkeley parquet ────────────────────────────────
print("\n  [B1/4] Loading measuring-hate-speech.parquet...")
df_b = pd.read_parquet(BERKELEY_PATH)
print(f"         Raw shape  : {df_b.shape}")
print(f"         Columns    : {list(df_b.columns)}\n")

# ── B2. Identify correct column names ────────────────────────
# Berkeley dataset has multiple annotators per comment
# We average scores per comment to get one row per text
print("  [B2/4] Aggregating multi-annotator scores per comment...")

# Find text column
text_col = None
for candidate in ['text', 'comment', 'comment_text', 'sentence']:
    if candidate in df_b.columns:
        text_col = candidate
        break
if text_col is None:
    text_col = df_b.columns[0]
    print(f"         Warning: guessing text column = '{text_col}'")
else:
    print(f"         Text column    : '{text_col}'")

# Find hate speech score column
hate_col = None
for candidate in ['hatespeech', 'hate_speech_score', 'hate_speech', 'label']:
    if candidate in df_b.columns:
        hate_col = candidate
        break
if hate_col is None:
    print("         Warning: hatespeech column not found — using first numeric")
    hate_col = df_b.select_dtypes(include=[np.number]).columns[0]
print(f"         Hate score col : '{hate_col}'")

# Find sentiment column for sarcasm
sent_col = None
for candidate in ['sentiment', 'irony', 'sarcasm']:
    if candidate in df_b.columns:
        sent_col = candidate
        break
print(f"         Sentiment col  : '{sent_col}'")

# Aggregate: one row per unique text (mean of all annotator scores)
agg_dict = {hate_col: 'mean'}
if sent_col:
    agg_dict[sent_col] = 'mean'

# Check if there's a comment_id or unique text grouping column
id_col = None
for candidate in ['comment_id', 'id', 'annotator_id']:
    if candidate in df_b.columns:
        id_col = candidate
        break

if id_col and id_col != 'annotator_id':
    df_b_agg = df_b.groupby(id_col).agg(
        {text_col: 'first', **agg_dict}
    ).reset_index()
else:
    # Group by text directly
    df_b_agg = df_b.groupby(text_col).agg(agg_dict).reset_index()

print(f"         After aggregation: {df_b_agg.shape}")
df_b_agg = df_b_agg.dropna(subset=[text_col, hate_col])
df_b_agg['text_clean'] = df_b_agg[text_col].apply(clean_text)
df_b_agg = df_b_agg[df_b_agg['text_clean'].str.strip() != '']
print(f"         After cleaning   : {df_b_agg.shape}")

# ── B3. SBERT encode Berkeley texts ──────────────────────────
print(f"\n  [B3/4] Encoding Berkeley texts with SBERT...")
print(f"         {len(df_b_agg):,} unique comments to encode...")
t0 = time.time()
B_embeddings = sbert.encode(
    df_b_agg['text_clean'].tolist(),
    show_progress_bar=True,
    batch_size=32
)
print(f"         Done in {(time.time()-t0)/60:.1f} mins")

# ── B4. Train severity regressor ─────────────────────────────
print("\n  [B4/4] Training Berkeley models...")

# ── Severity: normalize hate score to 0-10 ───────────────────
hate_scores = df_b_agg[hate_col].to_numpy().astype(float)
# Normalize to 0-10 range
h_min, h_max = hate_scores.min(), hate_scores.max()
severity_target = ((hate_scores - h_min) / (h_max - h_min)) * 10
severity_target = np.clip(severity_target, 0, 10)

print(f"\n    → Severity Regressor...")
print(f"       Score range: {hate_scores.min():.2f} to {hate_scores.max():.2f}")
print(f"       Mapped to  : 0.0 to 10.0")

X_sev_tr, X_sev_te, y_sev_tr, y_sev_te = train_test_split(
    B_embeddings, severity_target,
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

sev_model = Ridge(alpha=1.0)
sev_model.fit(X_sev_tr, y_sev_tr)
sev_preds = np.clip(sev_model.predict(X_sev_te), 0, 10)
sev_mae   = mean_absolute_error(y_sev_te, sev_preds)
sev_r2    = r2_score(y_sev_te, sev_preds)
joblib.dump(sev_model, 'severity_model.pkl')
joblib.dump({'min': h_min, 'max': h_max}, 'severity_scaler.pkl')
print(f"       MAE : {sev_mae:.3f} (lower is better)")
print(f"       R²  : {sev_r2:.3f} (closer to 1.0 is better)")

# ── Sarcasm/implicit: negative sentiment = likely sarcasm ────
print(f"\n    → Sarcasm / Implicit Classifier...")

if sent_col and sent_col in df_b_agg.columns:
    sent_scores = df_b_agg[sent_col].to_numpy().astype(float)
    # Sarcastic/implicit = high hate score BUT positive/neutral sentiment
    # i.e. text that seems okay on surface but is rated hateful
    # Label: 1 = sarcastic/implicit, 0 = explicit/not hateful
    sarcasm_labels = (
        (hate_scores > np.percentile(hate_scores, 60)) &
        (sent_scores >= np.percentile(sent_scores, 40))
    ).astype(int)
    print(f"       Sarcasm positives: {sarcasm_labels.sum():,} / {len(sarcasm_labels):,}")
else:
    # Fallback: use hate score alone — borderline cases as implicit
    print("       Sentiment column not found — using hate score variance")
    sarcasm_labels = (
        (hate_scores > np.percentile(hate_scores, 50)) &
        (hate_scores < np.percentile(hate_scores, 80))
    ).astype(int)
    print(f"       Implicit positives: {sarcasm_labels.sum():,} / {len(sarcasm_labels):,}")

X_sar_tr, X_sar_te, y_sar_tr, y_sar_te = train_test_split(
    B_embeddings, sarcasm_labels,
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

sar_model = LogisticRegression(max_iter=1000, C=1.0,
                                random_state=RANDOM_STATE,
                                class_weight='balanced')
sar_model.fit(X_sar_tr, y_sar_tr)
sar_preds = sar_model.predict(X_sar_te)
sar_acc   = accuracy_score(y_sar_te, sar_preds)
joblib.dump(sar_model, 'sarcasm_model.pkl')
print(f"       Accuracy: {sar_acc:.4f}")
print(classification_report(
    y_sar_te, sar_preds,
    target_names=['Explicit/Safe', 'Implicit/Sarcastic']
))

# ════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TRAINING COMPLETE — FULL SUMMARY")
print("="*60)

print("\n  PART A — Classification Accuracy:")
for name, acc in sorted(results_cls.items(), key=lambda x: -x[1]):
    bar = "█" * int(acc * 40)
    print(f"    {name:<25} {acc:.4f}  {bar}")

print(f"\n  PART B — Berkeley Models:")
print(f"    Severity Regressor    MAE={sev_mae:.3f}  R²={sev_r2:.3f}")
print(f"    Sarcasm Classifier    Accuracy={sar_acc:.4f}")

print(f"\n  Training Data:")
print(f"    Twitter dataset       : {len(df):,} rows (with augmentation)")
print(f"    Berkeley dataset      : {len(df_b_agg):,} unique comments")
print(f"    SBERT model           : {SBERT_MODEL} (768-dim)")

print("\n  Output files:")
files = [
    'label_encoder.pkl', 'model_xgboost.pkl', 'model_svm.pkl',
    'model_lr.pkl', 'severity_model.pkl', 'severity_scaler.pkl',
    'sarcasm_model.pkl', 'X_test_embeddings.pkl', 'y_test.pkl',
    'sbert_embeddings_train.pkl', 'sbert_embeddings_test.pkl'
]
for f in files:
    size = os.path.getsize(f)/(1024*1024) if os.path.exists(f) else 0
    status = "✓" if os.path.exists(f) else "✗"
    print(f"    {status} {f:<42} {size:.1f} MB")

print("\n  Run: streamlit run app.py")
print("="*60 + "\n")