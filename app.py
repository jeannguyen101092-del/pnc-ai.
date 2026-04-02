import streamlit as st
import io, torch, pdfplumber, fitz
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from supabase import create_client

# ====== CONFIG ======
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(layout="wide", page_title="AI CLOUD PRO", page_icon="🛡️")

# ====== AI ======
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ====== PARSE ======
def parse_val(t):
    try:
        return float(t)
    except:
        return None

def get_data(pdf):
    specs = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf)) as p:
            for page in p.pages:
                table = page.extract_table()
                if table:
                    for r in table:
                        r = [str(x) for x in r if x]
                        if len(r) >= 2:
                            val = parse_val(r[-1])
                            if val:
                                specs[r[0]] = val

        doc = fitz.open(stream=pdf, filetype="pdf")
        img = Image.open(io.BytesIO(doc.load_page(0).get_pixmap().tobytes("png")))

        return {"spec": specs, "img": img}
    except:
        return None

# ====== LOAD DB ======
@st.cache_data(ttl=60)
def load_db():
    data = supabase.table("ai_data").select("*").execute()
    return data.data

# ====== UI ======
st.title("🛡️ AI MASTER PRO CLOUD")

# ===== Upload =====
files = st.file_uploader("📤 Upload dữ liệu", type="pdf", accept_multiple_files=True)

if files:
    for f in files:
        data = get_data(f.read())
        if data:
            with torch.no_grad():
                vec = ai_brain(tf(data['img'].convert('RGB')).unsqueeze(0)).flatten().cpu().numpy()

            supabase.table("ai_data").insert({
                "name": f.name,
                "vector": vec.tolist(),
                "spec": data['spec']
            }).execute()

    st.success("✅ Đã lưu cloud")

db = load_db()
st.write(f"📦 Tổng mẫu: {len(db)}")

# ===== Compare =====
up = st.file_uploader("🔍 File so sánh", type="pdf")

if up:
    target = get_data(up.read())

    if target:
        with torch.no_grad():
            t_vec = ai_brain(tf(target['img'].convert('RGB')).unsqueeze(0)).flatten().cpu().numpy()

        results = []
        excel = []

        for d in db:
            sim = float(cosine_similarity(
                t_vec.reshape(1,-1),
                np.array(d["vector"]).reshape(1,-1)
            )) * 100

            if sim > 60:
                results.append((d["name"], sim))
                excel.append({"File": d["name"], "Similarity": sim})

        for r in sorted(results, key=lambda x: x[1], reverse=True):
            st.write(f"{r[0]} - {round(r[1],1)}%")

        if excel:
            df = pd.DataFrame(excel)
            st.download_button(
                "📥 Xuất Excel",
                df.to_csv(index=False).encode(),
                "result.csv"
            )
