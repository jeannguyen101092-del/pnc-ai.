import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Auditor V48", page_icon="🔥")

# ================= MODEL =================
@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Model lỗi: {e}")
        return None

model_ai = load_ai()

# ================= PARSE =================
def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').lower()
        match = re.findall(r'\d+\.?\d*', txt)
        return float(match[0]) if match else 0
    except:
        return 0

# ================= EXTRACT =================
def extract_pom(pdf_file):
    specs, img_bytes = {}, None

    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(doc):
        img_bytes = doc.load_page(0).get_pixmap().tobytes("png")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue

                for i in range(len(df)):
                    k = str(df.iloc[i][0]).upper()
                    v = parse_val(df.iloc[i][1])
                    if v > 0:
                        specs[k] = v

    return {"specs": specs, "img": img_bytes}

# ================= VECTOR =================
def get_vector(img_bytes):
    try:
        if not img_bytes or not model_ai:
            return None

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0))
            v = v.view(-1).numpy()
            v = v / np.linalg.norm(v)

        return v.tolist()
    except:
        return None

# ================= SPEC SIM =================
def spec_similarity(spec1, spec2):
    keys = set(spec1) & set(spec2)
    if not keys:
        return 0

    diffs = []
    for k in keys:
        diffs.append(abs(spec1[k] - spec2[k]))

    avg_diff = np.mean(diffs)
    return max(0, 100 - avg_diff)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 DATA")

    try:
        res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Records", res.count)
    except:
        st.error("DB lỗi")

    files = st.file_uploader("Upload Techpack", accept_multiple_files=True)

    if files and st.button("🚀 NẠP"):
        for f in files:
            d = extract_pom(f)

            if not d['specs']:
                st.warning(f"Lỗi POM: {f.name}")
                continue

            vec = get_vector(d['img'])

            if vec is None:
                st.warning(f"⚠️ No vector: {f.name}")

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": d['specs']
            }).execute()

            st.success(f"✅ {f.name}")

        st.rerun()

# ================= MAIN =================
st.title("🔥 AI Auditor V48 – Hybrid")

file_test = st.file_uploader("Upload file kiểm", type="pdf")

if file_test:
    target = extract_pom(file_test)

    if target['specs']:
        st.success(f"{len(target['specs'])} POM")

        vec_test = get_vector(target['img'])

        db = supabase.table("ai_data").select("*").execute()

        results = []

        for row in db.data:
            sim_img = 0
            sim_spec = 0

            # AI
            if vec_test and row.get("vector"):
                v1 = np.array(vec_test).reshape(1,-1)
                v2 = np.array(row["vector"]).reshape(1,-1)
                sim_img = cosine_similarity(v1, v2)[0][0] * 100

            # SPEC
            sim_spec = spec_similarity(target['specs'], row['spec_json'])

            # HYBRID
            sim = 0.4 * sim_img + 0.6 * sim_spec

            results.append({
                "file": row["file_name"],
                "sim": sim,
                "img": sim_img,
                "spec": sim_spec,
                "data": row
            })

        top = sorted(results, key=lambda x: x['sim'], reverse=True)[:3]

        for r in top:
            st.subheader(f"{r['file']} → {r['sim']:.1f}%")
            st.write(f"AI: {r['img']:.1f}% | SPEC: {r['spec']:.1f}%")

            diff = []
            for k, v in target['specs'].items():
                ref = r['data']['spec_json'].get(k, 0)
                diff.append([k, v, ref, round(v-ref,2)])

            df = pd.DataFrame(diff, columns=["POM","NEW","REF","DIFF"])
            st.dataframe(df)

# ================= RESET =================
if st.button("RESET"):
    st.rerun()
