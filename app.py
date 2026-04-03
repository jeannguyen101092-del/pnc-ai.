# ==========================================================
# AI FASHION PRO V8 (CLEAN SPEC + FILTER RÁC)
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8", page_icon="👔")

# ================= AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ================= PARSE VALUE =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        return 0

# ================= FILTER KEY =================
VALID_KEYS = [
    'INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN',
    'CHEST','LENGTH','SLEEVE','SHOULDER','BOTTOM'
]

BLOCK_KEYS = [
    'SIZE','SEASON','TECH','DATE','#','DEVELOPMENT',
    'FABRIC','BODY','SHELL','LINING','MATERIAL',
    'COTTON','POLYESTER','ELASTANE','NYLON',
    '%','PFD','DYED','WASH','COLOR','PRINT'
]

# ================= PARSER V8 =================
def extract_specs(table):
    specs = {}

    for r in table:
        if not r or len(r) < 2:
            continue

        row_text = " | ".join([str(x) for x in r if x]).upper()

        # ❌ bỏ rác
        if any(x in row_text for x in BLOCK_KEYS):
            continue

        # ❌ không phải POM kỹ thuật
        if not any(k in row_text for k in VALID_KEYS):
            continue

        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]

        if len(vals) < 1:
            continue

        val = float(np.median(vals))

        key = row_text[:120]

        specs[key] = round(val, 2)

    return specs

# ================= CLASSIFY =================
def advanced_classify(specs, text, file_name):
    txt = (text + " " + file_name).upper()
    inseam = specs.get('INSEAM', 0)

    if 'BIB' in txt:
        return "QUẦN YẾM"

    if 'CARGO' in txt:
        return "QUẦN CARGO"

    if 'ELASTIC' in txt:
        return "QUẦN LƯNG THUN"

    if inseam > 0:
        if inseam <= 11:
            return "QUẦN SHORT"
        if inseam >= 25:
            return "QUẦN DÀI"

    if 'DRESS' in txt:
        return "ĐẦM"
    if 'SKIRT' in txt:
        return "VÁY"
    if 'SHIRT' in txt:
        return "ÁO SƠ MI"

    return "ÁO"

# ================= GET DATA =================
def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""

        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    all_texts += t + " "

                tables = p.extract_tables()
                for table in tables:
                    specs.update(extract_specs(table))

        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        if len(specs) < 5:
            return None

        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))

        return {
            "spec": specs,
            "img_b64": img_b64,
            "img_bytes": img_bytes,
            "cat": cat
        }

    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

# ================= UPLOAD =================
with st.sidebar:
    st.header("📦 NẠP KHO")

    files = st.file_uploader("Upload PDF", accept_multiple_files=True)

    if files and st.button("🚀 NẠP"):
        for f in files:
            try:
                name = re.sub(r'\\s*\\(\\d+\\)', '', f.name)

                with open("tmp.pdf", "wb") as t:
                    t.write(f.getbuffer())

                d = get_data("tmp.pdf")

                if not d:
                    continue

                tf = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                with torch.no_grad():
                    vec = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()

                supabase.table("ai_data").upsert({
                    "file_name": name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "img_base64": d['img_b64'],
                    "category": d['cat']
                }, on_conflict="file_name").execute()

                os.remove("tmp.pdf")

            except Exception as e:
                st.warning(f"Lỗi {f.name}: {e}")

        st.success("✅ Done")
        st.rerun()

# ================= COMPARE =================
st.title("👔 AI Fashion Pro V8")

file = st.file_uploader("Upload test", type="pdf")

if file:
    with open("test.pdf", "wb") as f:
        f.write(file.getbuffer())

    target = get_data("test.pdf")

    if target:
        st.success(f"Nhận diện: {target['cat']}")

        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            results = []

            for i in db.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])[0][0]) * 100
                    results.append({
                        "name": i['file_name'],
                        "sim": sim,
                        "spec": i['spec_json'],
                        "img": i['img_base64']
                    })

            results = sorted(results, key=lambda x: x['sim'], reverse=True)[:10]

            for r in results:
                with st.expander(f"{r['name']} | {r['sim']:.1f}%"):
                    c1, c2 = st.columns(2)

                    with c1:
                        st.image(target['img_bytes'])
                    with c2:
                        st.image(base64.b64decode(r['img']))

                    diff = []
                    poms = set(target['spec']) | set(r['spec'])

                    for p in poms:
                        v1 = target['spec'].get(p, 0)
                        v2 = r['spec'].get(p, 0)
                        diff.append({
                            "POM": p,
                            "NEW": v1,
                            "OLD": v2,
                            "DIFF": round(v1 - v2, 2)
                        })

                    df = pd.DataFrame(diff)
                    st.dataframe(df)

                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)

                    st.download_button("📥 Excel", out.getvalue(), f"{r['name']}.xlsx")
