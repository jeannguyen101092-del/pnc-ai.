# AI FASHION AUDITOR V34 - FULL VERSION
# Upgraded: Image display, POM Description parsing, Auto size selection, tolerance compare
# NOTE: This is an expanded clean production-ready version

import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34", page_icon="📊")

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

# ================= VECTOR =================
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
            return model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except:
        return None

# ================= COMPARE =================
def compare_val(a,b,tol=0.5):
    try:
        return abs(float(a)-float(b))<=tol
    except:
        return False

# ================= EXTRACT =================
def extract_techpack(pdf_file):
    specs, img, raw = {}, None, ""

    try:
        pdf_bytes = pdf_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw += (page.extract_text() or "")
                tables = page.extract_tables() or []

                for tb in tables:
                    if not tb or len(tb)<2: continue

                    header = [str(h).upper() for h in tb[0]]

                    desc_idx = -1
                    for i,h in enumerate(header):
                        if "DESCRIPTION" in h:
                            desc_idx=i
                            break

                    if desc_idx==-1: continue

                    for row in tb[1:]:
                        if not row or len(row)<=desc_idx: continue

                        desc = str(row[desc_idx]).strip().upper()
                        values = [str(x) for i,x in enumerate(row) if i!=desc_idx and x]

                        if not values: continue

                        mid = len(values)//2
                        val = re.findall(r"\d+\.?\d*", values[mid])

                        if val:
                            specs[desc]=val[0]

        return {"spec":specs,"img":img}

    except:
        return None

# ================= LOAD DATA =================
def load_samples():
    try:
        res = supabase.table("ai_data").select("*").execute()
        return res.data if res.data else []
    except:
        return []

samples = load_samples()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("Kho dữ liệu")
    st.metric("Số mẫu", len(samples))

    files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

    if files and st.button("Nạp dữ liệu"):
        for f in files:
            d = extract_techpack(f)
            if not d: continue

            name = f.name.replace(".pdf","")

            img_path = f"{name}.png"
            img_url = None

            try:
                supabase.storage.from_(BUCKET).upload(
                    img_path,
                    d['img'],
                    file_options={"content-type":"image/png"},
                    upsert=True
                )
                img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
            except:
                pass

            vec = get_vector(d['img'])

            supabase.table("ai_data").upsert({
                "file_name":name,
                "vector":vec,
                "spec_json":d['spec'],
                "image_url":img_url
            }).execute()

        st.success("Upload xong")
        st.rerun()

# ================= MAIN =================
st.title("AI FASHION AUDITOR V34")

sample_list = ["AUTO AI"]+[s['file_name'] for s in samples]
selected = st.selectbox("Chọn mã gốc", sample_list)

files_test = st.file_uploader("Upload file test", type="pdf", accept_multiple_files=True)

if files_test:
    for f in files_test:
        data = extract_techpack(f)
        if not data: continue

        with st.expander(f.name, expanded=True):

            best=None
            best_score=0

            vt = get_vector(data['img'])

            if vt:
                vt = np.array(vt).reshape(1,-1)

                for s in samples:
                    try:
                        vr = s['vector']
                        if isinstance(vr,str): vr=json.loads(vr)
                        vs = np.array(vr).reshape(1,-1)

                        score = cosine_similarity(vt,vs)[0][0]

                        if score>best_score:
                            best_score=score
                            best=s
                    except:
                        continue

            if best:
                st.success(f"Match: {best['file_name']} | {best_score:.2%}")

                gspec = best.get('spec_json',{})
                if isinstance(gspec,str): gspec=json.loads(gspec)

                keys = set(list(data['spec'])+list(gspec))

                df = pd.DataFrame([
                    {
                        "POM":k,
                        "Test":data['spec'].get(k),
                        "Gốc":gspec.get(k),
                        "OK":compare_val(data['spec'].get(k), gspec.get(k))
                    }
                    for k in keys
                ])

                st.dataframe(df, use_container_width=True)

                # IMAGE
                st.divider()
                c1,c2 = st.columns(2)

                with c1:
                    st.subheader("Ảnh test")
                    st.image(Image.open(io.BytesIO(data['img'])), use_container_width=True)

                with c2:
                    st.subheader("Ảnh gốc")
                    if best.get("image_url"):
                        st.image(best['image_url'], use_container_width=True)
                    else:
                        st.warning("Chưa có ảnh")

            else:
                st.warning("Không tìm thấy mẫu")

# END FILE
