# ======================================================
# app.py 
# ======================================================

import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import classification_report, accuracy_score

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Phân tích hành vi người dùng trên MXH",
    layout="wide"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------- INIT SESSION STATE ----------
if "TEXT_COL" not in st.session_state:
    st.session_state.TEXT_COL = None
if "comments_input" not in st.session_state:
    st.session_state.comments_input = ""
if "comment_results" not in st.session_state:
    st.session_state.comment_results = []
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "topic_result" not in st.session_state:
    st.session_state.topic_result = None

# ---------- STOPWORDS & LEXICON ----------
VI_STOPWORDS = set("""
là thì mà rất quá này kia đó một các được cho với có những
đang đã sẽ cũng vì tại từ khi nếu nhưng nên
""".split())

NEGATION_WORDS = {"không", "chẳng"}
SENTIMENT_WORDS = {
    "tốt","xấu","tệ","ổn","hay","dở",
    "đẹp","kinh","chán","ghê","ok","kém"
}

# ---------- REGEX ----------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
NON_ALPHA_RE = re.compile(
    r'[^a-z0-9áàảãạăắằẳẵặâấầẩẫậ'
    r'éèẻẽẹêếềểễệ'
    r'íìỉĩị'
    r'óòỏõọôốồổỗộơớờởỡợ'
    r'úùủũụưứừửữự'
    r'ýỳỷỹỵđ ]'
)
MULTI_SPACE_RE = re.compile(r'\s+')

# ---------- CLEAN TEXT (TRAIN – GIỮ NGUYÊN LOGIC GỐC) ----------
def clean_text_sentiment(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r'\1', s)
    s = NON_ALPHA_RE.sub(" ", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()

    words = s.split()
    output, negate = [], False

    for w in words:
        if w in NEGATION_WORDS:
            negate = True
            continue
        if negate and w in SENTIMENT_WORDS:
            output.append("NEG_" + w)
        elif w not in VI_STOPWORDS and len(w) > 1:
            output.append(w)
        negate = False

    return " ".join(output)

# ---------- CLEAN TEXT (INFERENCE – ỔN ĐỊNH) ----------
def clean_text_inference(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r'\1', s)
    s = NON_ALPHA_RE.sub(" ", s)
    return MULTI_SPACE_RE.sub(" ", s).strip()

def clean_text_topic(s):
    return clean_text_inference(s)

# ---------- LABEL ----------
def sentiment_bucket(label):
    label = str(label).lower()
    if "neg" in label:
        return "Negative"
    if "pos" in label:
        return "Positive"
    return None

# ---------- CONFIDENCE ----------
def calibrated_confidence(p):
    return np.clip(0.5 + (p - 0.5) * 1.6, 0, 1)

def confidence_level(conf):
    if conf >= 75:
        return "Cao"
    if conf >= 60:
        return "Trung bình"
    return "Thấp"

def confidence_icon(level):
    return {"Cao": "🟢", "Trung bình": "🟡", "Thấp": "🔴"}[level]

def explain_comment(sentiment, conf):
    lvl = confidence_level(conf)
    if sentiment == "Positive":
        return f"Cảm xúc tích cực ({lvl}) – người dùng có xu hướng hài lòng."
    return f"Cảm xúc tiêu cực ({lvl}) – người dùng có dấu hiệu không hài lòng."

# ---------- LOAD DATA ----------
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, encoding="latin-1")

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Ứng dụng MXH")
uploaded = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
if not uploaded:
    st.stop()

df = load_data(uploaded)

page = st.sidebar.radio(
    "📌 Chức năng",
    [
        "📁 Xem dữ liệu",
        "🧠 Huấn luyện mô hình",
        "💬 Phân tích comment",
        "🛒 Phân tích theo sản phẩm",
        "🧩 Topic Modeling"
    ]
)

# ======================================================
# 📁 VIEW DATA
# ======================================================
if page == "📁 Xem dữ liệu":
    st.title("📁 Dữ liệu đầu vào")

    show_processed = st.checkbox(
        "🔍 Hiển thị dữ liệu đã xử lý (clean_text)",
        value=st.session_state.df_processed is not None
    )

    if show_processed:
        if st.session_state.TEXT_COL is None:
            st.warning("Cần huấn luyện model trước")
        else:
            if st.session_state.df_processed is None:
                tmp = df.copy()
                tmp["clean_text"] = tmp[
                    st.session_state.TEXT_COL
                ].apply(clean_text_sentiment)
                st.session_state.df_processed = tmp

            st.dataframe(
                st.session_state.df_processed[
                    [st.session_state.TEXT_COL, "clean_text"]
                ].head(50)
            )
    else:
        st.dataframe(df.head(50))

# ======================================================
# 🧠 TRAIN SENTIMENT
# ======================================================
elif page == "🧠 Huấn luyện mô hình":
    st.title("🧠 Huấn luyện Sentiment")

    text_col = st.selectbox("📄 Cột nội dung", df.columns)
    label_col = st.selectbox("🏷️ Cột nhãn", df.columns)

    if st.button("▶️ Huấn luyện"):
        st.session_state.TEXT_COL = text_col
        st.session_state.df_processed = None

        data = df[[text_col, label_col]].dropna().copy()
        data["clean"] = data[text_col].apply(clean_text_sentiment)
        data["label"] = data[label_col].apply(sentiment_bucket)
        data = data.dropna()

        Xtr, Xte, ytr, yte = train_test_split(
            data["clean"],
            data["label"],
            test_size=0.2,
            stratify=data["label"],
            random_state=42
        )

        model = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=14000,
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True
                )),
                ("char", TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 4),
                    max_features=2000
                ))
            ], transformer_weights={"word": 0.85, "char": 0.15})),
            ("clf", LogisticRegression(
                max_iter=500,
                solver="liblinear",
                class_weight="balanced"
            ))
        ])

        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))

        st.success(f"Accuracy: {acc:.2%}")
        st.text(classification_report(yte, model.predict(Xte)))

        joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")

# ======================================================
# 💬 COMMENT ANALYSIS
# ======================================================
elif page == "💬 Phân tích comment":
    st.title("💬 Phân tích comment")

    model = joblib.load(MODEL_DIR / "sentiment_model.joblib")

    comments = st.text_area(
        "✍️ Nhập comment (mỗi dòng = 1 comment)",
        height=200,
        value=st.session_state.comments_input
    )

    if st.button("🔍 Phân tích"):
        st.session_state.comments_input = comments

        rows = [c for c in comments.split("\n") if c.strip()]
        clean = [clean_text_inference(c) for c in rows]

        probs = model.predict_proba(clean)
        preds = model.predict(clean)
        confs = calibrated_confidence(
            np.max(probs, axis=1)
        ) * 100

        st.session_state.comment_results = list(zip(rows, preds, confs))

    for i, (raw, pred, conf) in enumerate(st.session_state.comment_results, 1):
        lvl = confidence_level(conf)
        st.markdown(f"### 💬 Comment {i}")
        st.write(raw)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sentiment", pred)
        with c2:
            st.metric("Độ tin cậy", f"{confidence_icon(lvl)} {lvl}")
        with c3:
            st.metric("Xác suất", f"{conf:.1f}%")

        st.info(explain_comment(pred, conf))
        st.divider()

# ======================================================
# 🛒 PRODUCT ANALYSIS (GIỮ NGUYÊN)
# ======================================================
elif page == "🛒 Phân tích theo sản phẩm":
    st.title("🛒 Phân tích phản hồi theo sản phẩm")

    model = joblib.load(MODEL_DIR / "sentiment_model.joblib")
    col = st.session_state.TEXT_COL
    keyword = st.text_input("🔎 Từ khóa sản phẩm")

    if st.button("📊 Phân tích") and col:
        dfp = df[
            df[col].astype(str).str.contains(keyword, case=False, na=False)
        ].copy()

        dfp["sentiment"] = model.predict(
            dfp[col].apply(clean_text_inference)
        )

        st.bar_chart(dfp["sentiment"].value_counts())
        st.dataframe(dfp[[col, "sentiment"]].head(50))

# ======================================================
# 🧩 TOPIC MODELING (CÓ NÚT CHẠY)
# ======================================================
elif page == "🧩 Topic Modeling":
    st.title("🧩 Topic Modeling")

    col = st.session_state.TEXT_COL
    if col is None:
        st.warning("Cần huấn luyện model trước")
        st.stop()

    algo = st.selectbox("Thuật toán", ["LDA", "NMF"])
    n_topics = st.slider("Số topic", 2, 12, 5)

    if st.button("▶️ Chạy"):
        texts = df[col].astype(str).apply(clean_text_topic)

        if algo == "LDA":
            vec = CountVectorizer(min_df=5)
            X = vec.fit_transform(texts)
            tm = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
        else:
            vec = TfidfVectorizer(min_df=5)
            X = vec.fit_transform(texts)
            tm = NMF(
                n_components=n_topics,
                random_state=42
            )

        tm.fit(X)
        words = vec.get_feature_names_out()

        topics = []
        for topic in tm.components_:
            top = topic.argsort()[::-1][:10]
            topics.append(
                [(words[i], topic[i]) for i in top]
            )

        st.session_state.topic_result = topics

    if st.session_state.topic_result:
        for i, topic in enumerate(st.session_state.topic_result, 1):
            st.subheader(f"📌 Topic {i}")
            st.dataframe(
                pd.DataFrame(topic, columns=["Word", "Importance"])
            )
