import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. CẤU HÌNH (Thay URL và KEY của bạn)
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI SMART SPEC PRO V2.1", page_icon="👔")

if 'sel_code' not in st.session_state: st.session_state.sel_code = None

# --- HÀM HỖ TRỢ AI & PHÂN LOẠI THÔNG MINH ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

def detect_category(text_list, file_name):
    txt = (" ".join(text_list) + " " + file_name).upper()
    
    # 1. Ưu tiên nhận diện QUẦN (Để tránh nhầm với Jacket)
    pant_keys = ['PANT', 'TROUSER', 'JEAN', 'SHORT', 'LEGGIN', 'BOTTOM', 'INSEAM', 'OUTSEAM', 'CROTCH', 'THIGH', 'HIP', 'WAIST']
    if any(k in txt for k in pant_keys): return "QUẦN"
    
    # 2. Nhận diện VÁY/ĐẦM
    if any(k in txt for k in ['DRESS', 'SKIRT', 'GOWN']): return "VÁY/ĐẦM"
    
    # 3. Nhận diện JACKET/VEST
    if any(k in txt for k in ['JACKET', 'COAT', 'BOMBER', 'PARKA', 'HOODIE', 'VEST', 'BLAZER', 'SUIT']): return "JACKET/VEST"
    
    # 4. Nhận diện ÁO
    if any(k in txt for k in ['TEE', 'SHIRT', 'TOP', 'POLO', 'SWEATER', 'CHEST', 'BUST', 'ARMHOLE']): return "ÁO (TEE/SHIRT)"
    
    return "KHÁC"

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
    specs, all_texts = {}, []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: all_texts.append(t)
                for table in p.extract_tables():
                    for r in table:
                        if not r or len(r) < 2: continue
                        val = parse_val(r[-1])
                        # Lấy tên POM sạch hơn
                        pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                        if val > 0 and len(pom) > 3: specs[pom] = val
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        return {"spec": specs, "img": pix.tobytes("png"), "cat": detect_category(all_texts, os.path.basename(pdf_path))}
    except: return None

# --- GIAO DIỆN ---
st.title("👔 AI SMART SPEC PRO - QUẢN LÝ KHO THÔNG MINH")

with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    try:
        res_count = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng mẫu trong kho", res_count.count if res_count.count else 0)
    except: st.error("Lỗi kết nối database!")
    
    st.divider()
    up_bulk = st.file_uploader("📥 Nạp kho hàng loạt (PDF)", type="pdf", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP KHO"):
        bar = st.progress(0)
        for i, f in enumerate(up_bulk):
            with open("temp.pdf", "wb") as tmp: tmp.write(f.getbuffer())
            d = get_data("temp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v.tolist(), "spec_json": d['spec'], "category": d['cat']}).execute()
            bar.progress((i+1)/len(up_bulk))
        st.success("Đã nạp xong!")
        st.rerun()

# --- PHẦN SO SÁNH ---
st.subheader("🔍 SO SÁNH MẪU MỚI")
up_test = st.file_uploader("Tải file cần kiểm tra", type="pdf", key="tester")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.success(f"🎯 Hệ thống nhận diện đây là: **{target['cat']}**")
        
        # SỬA LỖI ĐỎ: Kiểm tra dữ liệu trước khi so sánh
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db_res.data:
            st.warning(f"Chưa có mẫu nào thuộc loại '{target['cat']}' trong kho để so sánh.")
        else:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            # Tính toán so sánh an toàn hơn
            sim_list = []
            for item in db_res.data:
                try:
                    vec_db = np.array(item['vector'], dtype=np.float32)
                    sim = float(cosine_similarity([v_test], [vec_db])) * 100
                    sim_list.append({"name": item['file_name'], "sim": sim, "spec": item['spec_json']})
                except: continue
            
            sim_list = sorted(sim_list, key=lambda x: x['sim'], reverse=True)[:4]
            
            if sim_list:
                if st.session_state.sel_code is None: st.session_state.sel_code = sim_list[0]['name']

                cols = st.columns(4)
                for i, item in enumerate(sim_list):
                    with cols[i]:
                        st.info(f"📄 {item['name']}")
                        st.write(f"Khớp: **{item['sim']:.1f}%**")
                        if st.button("CHỌN MÃ", key=f"btn_{item['name']}"):
                            st.session_state.sel_code = item['name']
                            st.rerun()

                # Bảng so sánh
                st.divider()
                ref = next((x for x in sim_list if x['name'] == st.session_state.sel_code), sim_list[0])
                st.subheader(f"📊 BẢNG SO SÁNH VỚI: {st.session_state.sel_code}")
                
                diffs = []
                all_poms = sorted(list(set(target['spec'].keys()) | set(ref['spec'].keys())))
                for p in all_poms:
                    v1, v2 = target['spec'].get(p, 0), ref['spec'].get(p, 0)
                    diff = round(v1 - v2, 3)
                    status = "✅ KHỚP" if abs(diff) < 0.2 else ("⚠️ LỆCH" if abs(diff) < 1 else "❌ SAI BIỆT")
                    diffs.append({"POM": p, "File Mới": v1, "Mẫu Kho": v2, "Chênh lệch": diff, "Kết quả": status})
                
                df = pd.DataFrame(diffs)
                def color_row(row):
                    if row['Kết quả'] == "✅ KHỚP": return ['background-color: #d4edda']*len(row)
                    if row['Kết quả'] == "⚠️ LỆCH": return ['background-color: #fff3cd']*len(row)
                    return ['background-color: #f8d7da']*len(row)

                st.table(df.style.apply(color_row, axis=1))

                # Xuất Excel
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    df.to_excel(wr, index=False, sheet_name='Result')
                st.download_button("📥 TẢI EXCEL SO SÁNH", out.getvalue(), f"So_sanh_{up_test.name}.xlsx")
