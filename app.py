# ================= AI FASHION AUDITOR V35 PRO =================
# Upgrade: UI đẹp, so sánh AI mạnh hơn, tìm kiếm tương đồng, progress bar

import streamlit as st
import io, json, re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V35 PRO", page_icon="🧠")

# ================= INIT =================
@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

supabase = init_supabase()

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= IMAGE VECTOR =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
            return vec
    except:
        return None

# ================= LOAD DATA =================
def load_data():
    try:
        res = supabase.table("ai_data").select("*").execute()
        return res.data if res.data else []
    except:
        return []

# ================= SIMILARITY =================
def calc_similarity(v1, v2):
    try:
        return cosine_similarity([v1], [v2])[0][0]
    except:
        return 0

# ================= UI HEADER =================
st.markdown("""
<h1 style='text-align:center;color:#4CAF50;'>🧠 AI FASHION AUDITOR V35 PRO</h1>
<p style='text-align:center;'>So sánh & tìm kiếm mẫu tương đồng bằng AI</p>
""", unsafe_allow_html=True)

# ================= LOAD =================
data = load_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 Database")
    st.metric("Tổng mẫu", len(data))

    uploaded = st.file_uploader("Upload ảnh", type=["png","jpg","jpeg"])

    if uploaded:
        st.image(uploaded, caption="Ảnh upload", use_container_width=True)

# ================= MAIN =================
col1, col2 = st.columns([1,1])

query_vec = None

with col1:
    st.subheader("📥 Query")
    if uploaded:
        img_bytes = uploaded.read()
        query_vec = get_vector(img_bytes)

        if query_vec is None:
            st.error("Không đọc được ảnh")

with col2:
    st.subheader("📊 Kết quả AI")

    if query_vec is not None and data:

        results = []

        for item in data:
            if not item.get("vector"):
                continue

            sim = calc_similarity(query_vec, item["vector"])

            results.append({
                "name": item["file_name"],
                "score": sim,
                "img": item.get("image_url")
            })

        # sort
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # top 5
        top_k = results[:5]

        for r in top_k:
            st.markdown("---")

            c1, c2 = st.columns([1,2])

            with c1:
                if r["img"]:
                    st.image(r["img"], use_container_width=True)

            with c2:
                percent = round(r["score"]*100,2)

                st.markdown(f"### {r['name']}")

                st.progress(r["score"])

                st.success(f"Độ tương đồng: {percent}%")

# ================= SEARCH FILTER =================
st.markdown("---")
st.subheader("🔎 Tìm kiếm nâng cao")

keyword = st.text_input("Nhập tên mẫu")

if keyword:
    filtered = [d for d in data if keyword.lower() in d["file_name"].lower()]

    st.write(f"Tìm thấy {len(filtered)} mẫu")

    for f in filtered:
        st.write(f["file_name"])

# ================= FOOTER =================
st.markdown("---")
st.caption("V35 PRO | AI Similarity Engine | ResNet50")
