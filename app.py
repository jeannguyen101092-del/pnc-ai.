# ==========================================================
# AI FASHION PRO V8.4 - FIX API ERROR & SMART CLASSIFY
# ==========================================================
import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
# Thay URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.4", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- CÔNG CỤ XỬ LÝ ---
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

# Từ khóa để nhận diện Quần chính xác hơn
PANT_KEYS = ['INSEAM', 'WAIST', 'HIP', 'RISE', 'THIGH', 'KNEE', 'LEG OPENING']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r or len(r) < 2: continue
        # Lấy cột đầu tiên làm Hạng mục (Description)
        desc = str(r[0]).strip().upper()
        if not desc or len(desc) < 3 or any(x in desc for x in ['DATE', 'SEASON', 'PAGE', 'FABRIC']): continue
        
        vals = [parse_val(x) for x in r[1:] if x]
        vals = [v for v in vals if v > 0]
        
        if vals:
            specs[desc] = round(float(np.median(vals)), 2)
    return specs

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
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        # NHẬN DIỆN THÔNG MINH: Ưu tiên từ khóa đặc trưng của Quần
        cat = "ÁO"
        text_full = (all_texts + " " + os.path.basename(pdf_path)).upper()
        if any(k in text_full for k in PANT_KEYS) or any(k in text_full for k in ['PANT', 'TROUSER', 'SHORT', 'JEAN']):
            cat = "QUẦN"
            if 'SHORT' in text_full: cat = "QUẦN SHORT"
            
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": cat}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            try:
                with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
                d = get_data("tmp.pdf")
                if d:
                    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        img_pill = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                        vec = ai_brain(tf(img_pill).unsqueeze(0)).flatten().numpy().tolist()
                    
                    # Nạp vào Supabase (Đã fix lỗi on_conflict)
                    supabase.table("ai_data").upsert({
                        "file_name": f.name,
                        "vector": vec,
                        "spec_json": d['spec'],
                        "img_base64": d['img_base64'],
                        "category": d['cat']
                    }, on_conflict="file_name").execute()
                os.remove("tmp.pdf")
            except Exception as e:
                st.warning(f"Lỗi file {f.name}: {e}")
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Đã nạp xong!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("👔 AI Fashion Pro V8.4")
test_file = st.file_uploader("Kéo thả file kiểm tra vào đây", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.info(f"Phân loại: {target['cat']} | Tìm thấy {len(target['spec'])} hạng mục POM")
        
        # Chỉ tìm mẫu cùng Category
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            matches = []
            for i in res.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:5]

            for m in matches:
                with st.expander(f"Khớp: {m['name']} ({m['sim']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img_bytes'], caption="File đang kiểm")
                    c2.image(base64.b64decode(m['img']), caption="Mẫu trong kho")

                    diff = []
                    # Đối soát hạng mục POM
                    for p, v1 in target['spec'].items():
                        # Tìm hạng mục tương ứng ở mẫu gốc
                        v2 = m['spec'].get(p, 0)
                        diff.append({
                            "Hạng mục (Description)": p,
                            "Thực tế": v1,
                            "Mẫu gốc": v2,
                            "Lệch": round(v1 - v2, 2),
                            "Kết quả": "✅ OK" if abs(v1-v2) <= 0.5 else "❌ LỆCH"
                        })
                    
                    df_res = pd.DataFrame(diff)
                    st.table(df_res)
                    
                    out = io.BytesIO()
                    df_res.to_excel(out, index=False)
                    st.download_button(f"📥 Tải báo cáo {m['name']}", out.getvalue(), f"Audit_{m['name']}.xlsx")
        else:
            st.warning("Kho dữ liệu chưa có mẫu tương ứng cho loại hàng này.")
