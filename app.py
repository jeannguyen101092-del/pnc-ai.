# ==========================================================
# AI FASHION V8.2 - GIỮ NGUYÊN UI + FIX SO SÁNH ĐÚNG LOẠI
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

st.set_page_config(layout="wide")

# ================= LOAD AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ================= PARSE VALUE =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\.?\d*)', str(t))
        return float(found[0]) if found else 0
    except:
        return 0

# ================= FILTER =================
VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG','CHEST','LENGTH','SLEEVE','SHOULDER']
BLOCK_KEYS = ['FABRIC','BODY','MATERIAL','COTTON','POLY','ELASTANE','%','COLOR']

# ================= EXTRACT SPEC =================
def extract_specs(table):
    specs = {}
    for r in table:
        row_text = " ".join([str(x) for x in r if x]).upper()

        if any(b in row_text for b in BLOCK_KEYS):
            continue

        if not any(v in row_text for v in VALID_KEYS):
            continue

        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]

        if vals:
            specs[row_text[:100]] = np.mean(vals)

    return specs

# ================= GET DATA =================
def get_data(pdf_path):
    try:
        specs, text = {}, ""

        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text += t
                for tb in p.extract_tables():
                    specs.update(extract_specs(tb))

        if len(specs) < 5:
            return None

        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        img = pix.tobytes("png")

        return {
            "spec": specs,
            "img": img
        }

    except:
        return None

# ================= UI =================
st.title("AI V8.2 FIX - SAME UI")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("temp.pdf","wb") as f:
        f.write(file.getbuffer())

    target = get_data("temp.pdf")

    if target:
        st.success("Đã đọc file")

        db = supabase.table("ai_data").select("*").execute()

        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        with torch.no_grad():
            v_test = ai_brain(tf(Image.open(io.BytesIO(target['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

        results = []

        for i in db.data:

            db_spec = i.get('spec_json', {})

            # ===== FIX SO SÁNH ĐÚNG LOẠI =====
            target_is_pant = any('INSEAM' in k for k in target['spec'])
            db_is_pant = any('INSEAM' in k for k in db_spec)

            target_is_top = any('CHEST' in k for k in target['spec'])
            db_is_top = any('CHEST' in k for k in db_spec)

            if target_is_pant != db_is_pant:
                continue

            if target_is_top != db_is_top:
                continue

            # ===== GIỮ NGUYÊN LOGIC CŨ =====
            if i.get('vector'):
                sim = float(cosine_similarity([v_test],[np.array(i['vector'])])[0][0])*100
                results.append({
                    "name": i['file_name'],
                    "sim": sim,
                    "spec": i['spec_json'],
                    "img": i['img_base64']
                })

        results = sorted(results, key=lambda x:x['sim'], reverse=True)[:10]

        for r in results:
            with st.expander(f"{r['name']} | {r['sim']:.1f}%"):
                st.image(target['img'])
                st.image(base64.b64decode(r['img']))

                diff=[]
                poms=set(target['spec'])|set(r['spec'])
                for p in poms:
                    diff.append({
                        "POM":p,
                        "NEW":target['spec'].get(p,0),
                        "OLD":r['spec'].get(p,0)
                    })

                st.dataframe(pd.DataFrame(diff))
