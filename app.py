# ==========================================================
# AI FASHION PRO V8.1 - FIXED API ERROR & POM
# ==========================================================
import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
# Điền URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.1", page_icon="👔")

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
        t_str = str(t).replace(',', '.')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= POM FILTER =================
# Mở rộng để dễ khớp hơn với Techpack Express
VALID_KEYS = ['WAIST', 'HIP', 'INSEAM', 'THIGH', 'KNEE', 'LEG', 'RISE', 'LENGTH', 'CHEST', 'SHOULDER']
BLOCK_KEYS = ['DATE', 'SEASON', 'FABRIC', 'MATERIAL', 'COLOR', 'PRINT', 'PAGE']

def extract_specs(table):
    specs = {}
    if not table: return specs
    for r in table:
        if not r or len(r) < 2: continue
        row_text = " ".join([str(x) for x in r if x]).upper()
        if any(b in row_text for b in BLOCK_KEYS): continue
        
        # Nếu dòng chứa từ khóa kỹ thuật hoặc có ít nhất 2 con số (size)
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        
        if len(vals) >= 1 and any(k in row_text for k in VALID_KEYS):
            # Lấy cột đầu làm mô tả, trung vị các số làm giá trị đại diện
            key = str(r[0]).strip().upper()[:100]
            specs[key] = round(float(np.median(vals)), 2)
    return specs

# ================= CLASSIFY =================
def advanced_classify(specs, text, file_name):
    txt = (str(text) + " " + str(file_name)).upper()
    if 'INSEAM' in specs or 'WAIST' in specs:
        if 'SHORT' in txt: return "QUẦN SHORT"
        return "QUẦN"
    if 'DRESS' in txt or 'SKIRT' in txt: return "VÁY/ĐẦM"
    return "ÁO"

# ================= GET DATA =================
def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                all_texts += (p.extract_text() or "") + " "
                tables = p.extract_tables()
                for table in tables:
                    specs.update(extract_specs(table))
        
        doc = fitz.open(pdf_path)
        # Lấy ảnh trang 1 hoặc trang có Sketch
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": cat}
    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

# ================= SIDEBAR: NẠP KHO =================
with st.sidebar:
    st.header("📦 NẠP KHO")
    files = st.file_uploader("Upload PDF mẫu gốc", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    img_pill = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                    vec = ai_brain(tf(img_pill).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "img_base64": d['img_b64'],
                    "category": d['cat']
                }, on_conflict="file_name").execute()
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Đã nạp xong!")
        st.rerun()

# ================= MAIN: KIỂM TRA =================
st.title("👔 AI Fashion Pro V8.1")
test_file = st.file_uploader("Upload file cần kiểm tra", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")

    if target:
        st.info(f"Phân loại: {target['cat']} | Tìm thấy {len(target['spec'])} thông số POM")
        
        # FIX LỖI: Kiểm tra database
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            results = []
            for i in res.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])[0][0]) * 100
                    results.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            results = sorted(results, key=lambda x: x['sim'], reverse=True)[:5]

            for r in results:
                with st.expander(f"Mẫu khớp: {r['name']} ({r['sim']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img_bytes'], caption="Mẫu kiểm tra")
                    c2.image(base64.b64decode(r['img']), caption="Mẫu gốc")

                    diff = []
                    # So sánh các hạng mục POM chung
                    for p in target['spec']:
                        v1 = target['spec'][p]
                        # Tìm hạng mục tương tự ở mẫu gốc
                        v2 = next((r['spec'][k] for k in r['spec'] if p in k or k in p), 0)
                        diff.append({"Hạng mục (POM)": p, "Mẫu kiểm": v1, "Mẫu gốc": v2, "Chênh lệch": round(v1 - v2, 2)})

                    df_res = pd.DataFrame(diff)
                    st.table(df_res)
                    
                    excel = io.BytesIO()
                    df_res.to_excel(excel, index=False)
                    st.download_button("📥 Tải báo cáo Excel", excel.getvalue(), f"Audit_{r['name']}.xlsx")
        else:
            st.warning("Không tìm thấy mẫu cùng loại trong kho dữ liệu.")
