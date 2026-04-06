# ================= FULL CODE V26 (GIỮ NGUYÊN UI + FIX LOGIC + THÊM CHỌN MÃ) =================
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI POM PRO V26", page_icon="🛡️")

# ================= AI =================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai = load_model()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        with torch.no_grad():
            return ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except:
        return None

# ================= EXTRACT =================
def extract(pdf_file):
    specs, img = {}, None
    pdf_bytes = pdf_file.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()
    except:
        pass

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for tb in page.extract_tables() or []:
                for row in tb:
                    row = [str(c).strip() for c in row if c and str(c).strip()]
                    if len(row) < 2: continue

                    key = " ".join(row[:-1]).upper()
                    val = row[-1]

                    if len(key) > 3 and any(c.isdigit() for c in val):
                        specs[key] = val

    if img and len(specs) >= 3:
        return {"spec": specs, "img": img}
    return None

# ================= LOAD DB =================
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
except:
    samples = []

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 KHO")
    st.write(f"Tổng: {len(samples)}")

    # 🔥 chọn mã thủ công
    list_ma = [s['file_name'] for s in samples]
    selected_ma = st.selectbox("Chọn mã so sánh", ["AUTO"] + list_ma)

    up = st.file_uploader("Upload PDF kho", type="pdf", accept_multiple_files=True)
    if up and st.button("Nạp kho"):
        for f in up:
            d = extract(f)
            if d:
                ma = f.name.replace(".pdf","")
                vec = get_vector(d['img'])
                supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                supabase.table("ai_data").upsert({
                    "file_name":ma,
                    "vector":vec,
                    "spec_json":d['spec'],
                    "img_url":url
                }).execute()
                st.success(ma)
        st.rerun()

# ================= MAIN =================
st.title("🛡️ AI POM PRO V26")

files = st.file_uploader("Upload PDF test", type="pdf", accept_multiple_files=True)

if files:
    for f in files:
        d = extract(f)
        if not d:
            st.error(f"Lỗi file: {f.name}")
            continue

        v_test = get_vector(d['img'])

        # ===== chọn mẫu =====
        best = None
        best_score = 0

        if selected_ma != "AUTO":
            best = next((s for s in samples if s['file_name']==selected_ma), None)
            best_score = 1.0
        else:
            for s in samples:
                if not s.get('vector'): continue
                score = float(cosine_similarity([v_test],[s['vector']])[0][0])
                if score > best_score:
                    best_score = score
                    best = s

        if not best:
            st.warning("Không tìm thấy")
            continue

        st.success(f"Match: {best['file_name']} ({best_score:.1%})")

        c1,c2 = st.columns(2)
        c1.image(d['img'])
        c2.image(best['img_url'])

        # ===== SO SÁNH =====
        res = []
        scores = []

        for kt, vt in d['spec'].items():
            best_k, best_r = None, 0

            for kg in best['spec_json'].keys():
                r = SequenceMatcher(None, kt, kg).ratio()
                if r > best_r:
                    best_r = r
                    best_k = kg

            vg = best['spec_json'].get(best_k, "---") if best_r>0.6 else "---"

            status = "OK" if str(vt)==str(vg) else "LECH"

            scores.append(best_r)

            res.append({
                "POM": kt,
                "Match": best_k,
                "%": round(best_r*100,1),
                "Test": vt,
                "Goc": vg,
                "KQ": status
            })

        df = pd.DataFrame(res)
        st.dataframe(df, use_container_width=True)

        if scores:
            st.info(f"Match tổng: {sum(scores)/len(scores)*100:.1f}%")

        st.download_button("Xuất Excel", df.to_excel(index=False), "compare.xlsx")
