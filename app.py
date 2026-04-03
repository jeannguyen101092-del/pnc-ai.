# ==========================================================
# AI FASHION PRO V8 - PARSER CHUẨN TECHPACK (GIỮ NGUYÊN SO SÁNH)
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# KẾT NỐI
# ==========================================
URL =  "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8", page_icon="👔")

# ==========================================
# AI MODEL
# ==========================================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ==========================================
# PARSE VALUE CHUẨN
# ==========================================
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

# ==========================================
# PARSER V8 - FIX TECHPACK
# ==========================================
def extract_specs(table):
    specs = {}
    for r in table:
        if not r or len(r) < 2:
            continue

        # 🔥 GHÉP FULL DÒNG (tránh mất nội dung dài)
        row_text = " | ".join([str(x) for x in r if x]).upper()

        # ❌ bỏ dòng rác
       if any(x in row_text for x in [
    'SIZE','SEASON','TECH','DATE','#','DEVELOPMENT',
    'FABRIC','BODY','SHELL','LINING','MATERIAL',
    'COTTON','POLYESTER','ELASTANE','NYLON',
    '%','PFD','DYED','WASH','COLOR','PRINT'
]):
    continue

        # 🔥 lấy tất cả số trong dòng (không phụ thuộc cột)
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]

        if len(vals) < 1:
            continue

        # 🔥 lấy giá trị đại diện (median ổn định hơn mean)
        val = float(np.median(vals))

        # 🔥 key = mô tả đầy đủ (không chỉ cột đầu)
        key = row_text[:120]  # giới hạn độ dài tránh quá dài

        if val > 0 and len(key) > 5:
            specs[key] = round(val, 2)

    return specs

# ==========================================
# CLASSIFY V8.2
# ==========================================
def advanced_classify(specs, text, file_name):
    txt = (text + " " + file_name).upper()
    inseam = specs.get('INSEAM', 0)

    if 'BIB' in txt:
        return "QUẦN YẾM"

    if 'CARGO' in txt or 'CARGO POCKET' in txt:
        return "QUẦN CARGO"

    if 'ELASTIC' in txt or 'WAISTBAND' in txt:
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

# ==========================================
# GET DATA
# ==========================================
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
            return None  # ❌ bỏ file lỗi

        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))

        return {
            "spec": specs,
            "img_b64": img_b64,
            "img_bytes": img_bytes,
            "cat": cat
        }

    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ==========================================
# SIDEBAR - NẠP KHO
# ==========================================
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO V8")

    up_bulk = st.file_uploader("Upload PDF", accept_multiple_files=True)

    if up_bulk and st.button("🚀 NẠP KHO"):
        for f in up_bulk:
            try:
                clean_name = re.sub(r'\\s*\\(\\d+\\)', '', f.name)

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
                    "file_name": clean_name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "img_base64": d['img_b64'],
                    "category": d['cat']
                }, on_conflict="file_name").execute()

                os.remove("tmp.pdf")

            except Exception as e:
                st.warning(f"Lỗi {f.name}: {e}")

        st.success("✅ Nạp kho hoàn tất")
        st.rerun()

# ==========================================
# SO SÁNH V8 - DROPDOWN MULTI RESULT
# ==========================================
st.title("👔 AI Fashion Pro V8")

up_test = st.file_uploader("Upload file test", type="pdf")

if up_test:
    with open("test.pdf", "wb") as f:
        f.write(up_test.getbuffer())

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

            st.subheader("📊 DANH SÁCH MẪU TƯƠNG ĐỒNG")

            for idx, r in enumerate(results):
                with st.expander(f"{r['name']}  |  🔥 {r['sim']:.1f}%"):
                    c1, c2 = st.columns(2)

                    with c1:
                        st.image(target['img_bytes'], caption="Mẫu mới", use_container_width=True)

                    with c2:
                        st.image(base64.b64decode(r['img']), caption=r['name'], use_container_width=True)

                    # bảng so sánh
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
                    st.dataframe(df, use_container_width=True)

                    # export riêng từng mã
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)

                    st.download_button(f"📥 Excel {r['name']}", out.getvalue(), f"{r['name']}.xlsx")
