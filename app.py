import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. KẾT NỐI (Thay URL và KEY của bạn)
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V5.1", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

# --- NÃO PHÂN LOẠI SIÊU CẤP ---
def advanced_classify(specs, text_content, file_name):
    txt = (text_content + " " + file_name).upper()
    inseam = specs.get('INSEAM', 0)
    
    # Ưu tiên nhận diện QUẦN trước để tránh nhầm với Jacket
    pant_keys = ['PANT', 'TROUSER', 'JEAN', 'SHORT', 'INSEAM', 'CROTCH', 'THIGH', 'HIP', 'WAIST']
    if any(k in txt for k in pant_keys) or inseam > 0:
        if 'BIB' in txt: return "QUẦN YẾM"
        if 'CARGO' in txt: return "QUẦN TÚI CARGO"
        return "QUẦN DÀI" if (inseam >= 25 or inseam == 0) else "QUẦN SHORT"

    if 'DRESS' in txt: return "ĐẦM"
    if 'SKIRT' in txt: return "VÁY"
    if any(k in txt for k in ['JACKET', 'COAT', 'VEST', 'BLAZER', 'SUIT']): return "JACKET/VEST"
    return "ÁO"

def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: all_texts += t + " "
                tables = p.extract_tables()
                for table in tables:
                    for r in table:
                        if r and len(r) >= 2:
                            val = parse_val(r[-1])
                            pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                            if val > 0 and len(pom) > 3: specs[pom] = val
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))
        return {"spec": specs, "img_b64": img_b64, "img_bytes": pix.tobytes("png"), "cat": cat}
    except: return None

# --- SIDEBAR: QUẢN TRỊ KHO ---
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_db.count if res_db.count else 0)
    
    up_bulk = st.file_uploader("Nạp file mẫu PDF", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in up_bulk:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v, "spec_json": d['spec'], "img_base64": d['img_b64'], "category": d['cat']}).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- PHẦN SO SÁNH ---
st.title("👔 AI Fashion Pro - Phân loại & Đối chiếu Thông minh")
up_test = st.file_uploader("📥 Tải file cần kiểm tra (PDF)", type="pdf")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.success(f"🎯 Hệ thống nhận diện: **{target['cat']}**")
        
        # Chỉ tìm trong kho cùng loại để so sánh
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db.data:
            st.warning(f"Kho chưa có mẫu nào thuộc loại '{target['cat']}' để đối chiếu.")
        else:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            sims = []
            for i in db.data:
                # CHẶN LỖI TYPEERROR: Chỉ so sánh nếu vector không rỗng
                if i.get('vector') is not None and len(i['vector']) > 0:
                    s = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    sims.append({"name": i['file_name'], "sim": s, "spec": i['spec_json'], "img": i['img_base64']})
            
            if sims:
                best = sorted(sims, key=lambda x: x['sim'], reverse=True)[0]
                c1, c2 = st.columns(2)
                with c1: st.image(target['img_bytes'], caption="Mẫu mới", use_container_width=True)
                with c2: st.image(base64.b64decode(best['img']), caption=f"Khớp nhất: {best['name']} ({best['sim']:.1f}%)", use_container_width=True)

                # BẢNG SO SÁNH 4 CỘT
                diff_list = []
                poms = sorted(list(set(target['spec'].keys()) | set(best['spec'].keys())))
                for p in poms:
                    v_n, v_o = target['spec'].get(p, 0), best['spec'].get(p, 0)
                    diff_list.append({"Thông số": p, "Mẫu Mới": v_n, "Mẫu Kho": v_o, "Chênh lệch": round(v_n - v_o, 2)})
                st.table(pd.DataFrame(diff_list))
                
                # NÚT EXCEL
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    pd.DataFrame(diff_list).to_excel(wr, index=False)
                st.download_button("📥 TẢI EXCEL SO SÁNH", out.getvalue(), "Ket_qua.xlsx")
