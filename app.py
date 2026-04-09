# ==========================================================
# AI FASHION PRO V37.6 - TURBO SCAN (PAGE 1 IMAGE + POM ONLY)
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- KẾT NỐI ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V37.6", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- TOOLS ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_specs_from_pom_page(table):
    specs = {}
    for r in table:
        if not r or len(r) < 2: continue
        # Cột 1 luôn là Hạng mục (Description)
        desc = str(r[0]).replace('\n', ' ').strip().upper()
        if len(desc) < 3 or any(x in desc for x in ['DATE', 'PAGE', 'TOTAL']): continue
        
        # Lấy giá trị số (thường là cột size trung tâm hoặc cột kế Description)
        vals = [parse_val(x) for x in r[1:] if x]
        vals = [v for v in vals if v > 0]
        if vals:
            specs[desc] = round(float(np.median(vals)), 2)
    return specs

def get_data_fast(pdf_file):
    try:
        specs, all_texts = {}, ""
        pdf_bytes = pdf_file.read()
        
        # 1. CHỈ QUÉT TRANG ĐẦU LẤY HÌNH ẢNH
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        # 2. CHỈ QUÉT TRANG CÓ CHỮ "POM" ĐỂ LẤY THÔNG SỐ
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").upper()
                # Bỏ qua nếu trang không chứa từ khóa POM hoặc SPEC
                if not any(k in text for k in ["POM", "SPECIFICATION", "MEASUREMENT", "TOLERANCE"]):
                    continue
                
                all_texts += text + " "
                tables = page.extract_tables()
                for table in tables:
                    specs.update(extract_specs_from_pom_page(table))
        
        if not specs: return None

        # Nhận diện Quần/Áo
        text_f = (all_texts + " " + pdf_file.name).upper()
        cat = "QUẦN" if any(k in text_f for k in ['WAIST','INSEAM','HIP','RISE','PANT','SHORT']) else "ÁO"
            
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": cat}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 NẠP KHO (FAST SCAN)")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = get_data_fast(f)
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    img_p = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                    "img_base64": d['img_b64'], "category": d['cat']
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Done!")
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Pro V37.6")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = get_data_fast(test_file)
    if target:
        st.info(f"Phân loại: {target['cat']} | POM: {len(target['spec'])}")
        
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                img_t = Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')
                v_test = ai_brain(tf(img_t).unsqueeze(0)).flatten().numpy()

            matches = []
            for i in res.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

            for m in matches:
                with st.expander(f"Khớp: {m['name']} ({m['sim']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img_bytes'], caption="File kiểm")
                    c2.image(base64.b64decode(m['img']), caption="Mẫu trong kho")

                    diff = []
                    for p, v1 in target['spec'].items():
                        v2 = m['spec'].get(p, 0)
                        # Khớp Description linh hoạt (8 ký tự)
                        if v2 == 0:
                            v2 = next((val for k, val in m['spec'].items() if p[:8] in k), 0)
                        
                        diff.append({"Hạng mục": p, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": round(v1-v2, 2), "Kết quả": "✅ OK" if abs(v1-v2)<=0.5 else "❌ LỆCH"})
                    
                    st.table(pd.DataFrame(diff))
    else:
        st.error("⚠️ Không thấy bảng POM. File có thể không có từ khóa 'POM' ở bất kỳ trang nào.")
