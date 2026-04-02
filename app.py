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

st.set_page_config(layout="wide", page_title="AI SMART SPEC PRO V2", page_icon="👔")

if 'sel_code' not in st.session_state: st.session_state.sel_code = None

# --- HÀM HỖ TRỢ AI & PHÂN LOẠI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

dedef detect_category(text_list, file_name):
    txt = (" ".join(text_list) + " " + file_name).upper()
    
    # Ưu tiên nhận diện QUẦN trước để tránh nhầm với Jacket (vì quần hay có từ khóa đặc thù)
    pant_keys = ['PANT', 'TROUSER', 'JEAN', 'SHORT', 'LEGGIN', 'BOTTOM', 'INSEAM', 'OUTSEAM', 'CROTCH', 'THIGH', 'HIP']
    if any(k in txt for k in pant_keys): return "QUẦN"
    
    # Tiếp theo là các loại đồ khác
    if any(k in txt for k in ['JACKET', 'COAT', 'BOMBER', 'PARKA', 'HOODIE']): return "JACKET/COAT"
    if any(k in txt for k in ['VEST', 'BLAZER', 'SUIT']): return "VEST/BLAZER"
    if any(k in txt for k in ['DRESS', 'SKIRT', 'GOWN']): return "VÁY/ĐẦM"
    if any(k in txt for k in ['TEE', 'SHIRT', 'TOP', 'POLO', 'SWEATER', 'CHEST', 'BUST']): return "ÁO (TEE/SHIRT)"
    
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
                        pom = str(r[0]).strip().upper()
                        if val > 0 and len(pom) > 3: specs[pom] = val
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        return {"spec": specs, "img": pix.tobytes("png"), "cat": detect_category(all_texts, os.path.basename(pdf_path))}
    except: return None

# --- GIAO DIỆN ---
st.title("👔 AI SMART SPEC PRO - HỆ THỐNG QUẢN LÝ KHO THÔNG MINH")

with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    res_count = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_count.count if res_count.count else 0)
    
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
        st.success(f"Đã nạp {len(up_bulk)} mẫu!")
        st.rerun()

# --- PHẦN SO SÁNH ---
st.subheader("🔍 SO SÁNH MẪU MỚI")
up_test = st.file_uploader("Tải file cần kiểm tra", type="pdf", key="tester")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.info(f"✅ Nhận diện: **{target['cat']}**")
        
        # Tìm trong kho cùng loại
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db_res.data:
            st.warning(f"Kho chưa có mẫu nào thuộc loại '{target['cat']}' để so sánh.")
        else:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            # Tính độ tương đồng
            sim_list = sorted([{"name": i['file_name'], "sim": float(cosine_similarity([v_test], [np.array(i['vector'])]))*100, "spec": i['spec_json']} for i in db_res.data], key=lambda x: x['sim'], reverse=True)[:4]
            
            if st.session_state.sel_code is None: st.session_state.sel_code = sim_list[0]['name']

            cols = st.columns(4)
            for i, item in enumerate(sim_list):
                with cols[i]:
                    st.write(f"📄 {item['name']}")
                    st.caption(f"Độ khớp: {item['sim']:.1f}%")
                    if st.button("CHỌN SO SÁNH", key=item['name']):
                        st.session_state.sel_code = item['name']
                        st.rerun()

            # Bảng so sánh chi tiết
            st.divider()
            ref = next(x for x in sim_list if x['name'] == st.session_state.sel_code)
            st.subheader(f"📊 KẾT QUẢ SO SÁNH VỚI: {st.session_state.sel_code}")
            
            diffs = []
            all_poms = sorted(list(set(target['spec'].keys()) | set(ref['spec'].keys())))
            for p in all_poms:
                v1, v2 = target['spec'].get(p, 0), ref['spec'].get(p, 0)
                diff = round(v1 - v2, 3)
                # Đánh giá mức độ lệch
                status = "✅ KHỚP" if abs(diff) < 0.2 else ("⚠️ LỆCH" if abs(diff) < 1 else "❌ SAI BIỆT")
                diffs.append({"Thông số (POM)": p, "Mẫu Mới": v1, "Mẫu Kho": v2, "Chênh lệch": diff, "Trạng thái": status})
            
            df = pd.DataFrame(diffs)
            
            # Tô màu bảng
            def color_diff(val):
                if val == "✅ KHỚP": return 'background-color: #d4edda'
                if val == "⚠️ LỆCH": return 'background-color: #fff3cd'
                return 'background-color: #f8d7da'
            
            st.table(df.style.applymap(color_diff, subset=['Trạng thái']))

            # Xuất Excel chuyên nghiệp
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Comparison')
                # Tự động căn chỉnh cột
                worksheet = writer.sheets['Comparison']
                for i, col in enumerate(df.columns):
                    worksheet.set_column(i, i, 20)
            
            st.download_button("📥 TẢI BÁO CÁO EXCEL CHUYÊN SÂU", output.getvalue(), f"Bao_cao_so_sanh_{up_test.name}.xlsx")
