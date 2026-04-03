# ==========================================================
# AI FASHION PRO V8.3 - FIX CLASSIFICATION + CLEAN SPEC
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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.3", page_icon="👔")

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

# ================= FILTER =================
VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG','CHEST','LENGTH','SLEEVE','SHOULDER']
BLOCK_KEYS = ['FABRIC','BODY','SHELL','LINING','MATERIAL','COTTON','POLY','ELASTANE','NYLON','%','COLOR','WASH']

# ================= PARSER =================
def extract_specs(table):
    specs = {}
    for r in table:
        if not r: continue
        row_text = " | ".join([str(x) for x in r if x]).upper()
        if any(b in row_text for b in BLOCK_KEYS): continue
        if not any(v in row_text for v in VALID_KEYS): continue
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        if not vals: continue
        key = row_text[:120]
        specs[key] = float(np.median(vals))
    return specs

# ================= CLASSIFY FIX =================
def advanced_classify(specs, text, file_name):
    txt = (text + " " + file_name).upper()
    score = {"ÁO": 0, "QUẦN": 0, "ĐẦM": 0, "VÁY": 0}

    for k in specs.keys():
        if any(x in k for x in ['CHEST','SLEEVE','SHOULDER']): score['ÁO'] += 2
        if 'INSEAM' in k or 'WAIST' in k or 'HIP' in k: score['QUẦN'] += 2

    if 'SHIRT' in txt: score['ÁO'] += 3
    if 'PANT' in txt or 'TROUSER' in txt: score['QUẦN'] += 3
    if 'DRESS' in txt: score['ĐẦM'] += 3
    if 'SKIRT' in txt: score['VÁY'] += 3

    best = max(score, key=score.get)

    if best == 'QUẦN':
        inseam = 0
        for k,v in specs.items():
            if 'INSEAM' in k:
                inseam = v
                break
        if 0 < inseam <= 11: return "QUẦN SHORT"
        if inseam >= 25: return "QUẦN DÀI"
        if 'CARGO' in txt: return "QUẦN CARGO"
        return "QUẦN"

    if best == 'ÁO':
        if 'SHIRT' in txt: return "ÁO SƠ MI"
        return "ÁO"
    return best

# ================= GET DATA =================
def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    specs.update(extract_specs(tb))

        if len(specs) < 5: return None

        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        img = pix.tobytes("png")

        return {
            "spec": specs,
            "img": img,
            "cat": advanced_classify(specs, text, os.path.basename(pdf_path))
        }
    except Exception as e:
        st.error(e)
        return None

# ================= UI =================
st.title("AI Fashion Pro V8.3")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("test.pdf","wb") as f:
        f.write(file.getbuffer())

    target = get_data("test.pdf")

    if target:
        st.success(f"Nhận diện: {target['cat']}")

        # Lấy dữ liệu từ kho
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
            # === CHỖ SỬA QUAN TRỌNG: CHỈ SO SÁNH NẾU CÙNG CHỦNG LOẠI (CAT) ===
            # Điều kiện này ép Áo so với Áo, Quần so với Quần đúng form
            db_cat = i.get('cat') # Lấy chủng loại đã lưu trong database
            
            if i.get('vector') and db_cat == target['cat']:
                sim = float(cosine_similarity([v_test],[np.array(i['vector'])])[0][0])*100
                results.append({
                    "name": i['file_name'],
                    "sim": sim,
                    "spec": i['spec_json'],
                    "img": i['img_base64']
                })

        # Sắp xếp và hiển thị
        results = sorted(results, key=lambda x:x['sim'], reverse=True)[:10]

        if not results:
            st.warning(f"Không tìm thấy mẫu nào cùng chủng loại {target['cat']} trong kho.")
        else:
            for r in results:
                with st.expander(f"{r['name']} | {r['sim']:.1f}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(target['img'], caption="Mẫu Test")
                    with col2:
                        st.image(base64.b64decode(r['img']), caption="Mẫu Đối Sánh")

                    diff=[]
                    poms=set(target['spec'])|set(r['spec'])
                    for p in poms:
                        diff.append({
                            "POM":p,
                            "NEW":target['spec'].get(p,0),
                            "OLD":r['spec'].get(p,0)
                        })
                    st.dataframe(pd.DataFrame(diff), use_container_width=True)
