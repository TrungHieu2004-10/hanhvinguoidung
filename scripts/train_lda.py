# scripts/train_lda.py

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================

# Thư mục gốc project (D:\hanhvinguoidung)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Đường dẫn file dữ liệu
# 👉 Nếu file dữ liệu của bạn KHÔNG tên là tweets.csv thì sửa ở đây
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "tweets.csv")

# Thư mục lưu model (đúng theo yêu cầu: D:\hanhvinguoidung\models)
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# HÀM TIỀN XỬ LÝ TEXT
# =========================

def clean_text(text: str) -> str:
    """
    Tiền xử lý một chuỗi văn bản:
    - chuyển về chữ thường
    - bỏ URL, @tag, #hashtag
    - bỏ ký tự đặc biệt
    - bỏ khoảng trắng thừa
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Bỏ URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Bỏ @username
    text = re.sub(r"@\w+", " ", text)

    # Bỏ dấu #
    text = text.replace("#", " ")

    # Bỏ ký tự đặc biệt, giữ chữ + số + tiếng Việt
    text = re.sub(
        r"[^0-9a-zA-Záàảãạăắằẳẵặâấầẩẫậ"
        r"éèẻẽẹêếềểễệíìỉĩị"
        r"óòỏõọôốồổỗộơớờởỡợ"
        r"úùủũụưứừửữự"
        r"ýỳỷỹỵđ\s]",
        " ",
        text,
    )

    # Bỏ khoảng trắng dư
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_series(series: pd.Series):
    """
    Nhận vào pandas Series chứa text → trả về list text đã xử lý.
    """
    series = series.dropna().astype(str)
    processed = series.apply(clean_text)
    processed = processed[processed.str.len() > 0]
    return processed.tolist()


# =========================
# HÀM HUẤN LUYỆN LDA
# =========================

def train_lda(texts, n_topics=6, max_features=2000):
    """
    Huấn luyện mô hình LDA từ list chuỗi 'texts'.
    Trả về:
      - mô hình LDA
      - vectorizer (CountVectorizer)
    """
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts phải là list/tuple các chuỗi văn bản đã tiền xử lý.")

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=max_features,
        stop_words="english",  # nếu dữ liệu thuần Việt là chính, có thể đổi None
    )

    X = vectorizer.fit_transform(texts)

    n_components = min(n_topics, X.shape[0])  # tránh n_topics > số mẫu
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_method="batch",
        random_state=42,
    )

    lda.fit(X)
    return lda, vectorizer


# =========================
# TỰ ĐỘNG ĐOÁN CỘT TEXT
# =========================

def detect_text_col(df: pd.DataFrame):
    """
    Tự động đoán cột văn bản trong DataFrame.
    Ưu tiên các tên quen thuộc, nếu không có thì chọn cột kiểu object đầu tiên.
    """
    candidates = ["text", "tweet", "content", "message", "body"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if df[c].dtype == object:
            return c

    return None


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # 1. Kiểm tra dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy file dữ liệu: {DATA_PATH}")
        print("👉 Hãy đặt file CSV vào data/raw/tweets.csv hoặc sửa lại DATA_PATH trong train_lda.py.")
        raise SystemExit(1)

    print(f"✅ Đang đọc dữ liệu từ: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2. Xác định cột text
    text_col = detect_text_col(df)
    if text_col is None:
        print("❌ Không tìm thấy cột văn bản trong CSV.")
        print("   Hãy kiểm tra lại tên cột và cập nhật hàm detect_text_col().")
        raise SystemExit(1)

    print(f"✅ Đã chọn cột văn bản: '{text_col}'")

    # 3. Tiền xử lý
    texts = preprocess_series(df[text_col])
    print(f"✅ Đã tiền xử lý {len(texts)} dòng văn bản.")

    if len(texts) < 10:
        print("⚠️ Dữ liệu sau tiền xử lý quá ít (< 10 dòng), LDA có thể không ổn.")

    # 4. Huấn luyện LDA
    print("⏳ Đang huấn luyện LDA...")
    lda, vect = train_lda(texts, n_topics=6, max_features=2000)
    print("✅ Huấn luyện xong LDA.")

    # 5. Lưu model
    lda_path = os.path.join(OUT_DIR, "lda_model.pkl")
    vec_path = os.path.join(OUT_DIR, "vectorizer.pkl")

    joblib.dump(lda, lda_path)
    joblib.dump(vect, vec_path)

    print("✅ Đã lưu mô hình:")
    print("   LDA model  ->", lda_path)
    print("   Vectorizer ->", vec_path)
