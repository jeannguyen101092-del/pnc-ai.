import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, random
from torchvision import models, transforms
from supabase import create_client, Client
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.49", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM HỖ TRỢ =================
def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        if not t_str or any(x in t_str.upper() for x in ['SIZE', 'TOL', 'DATE']): return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_data(pdf_path):
    try:
        specs, text, base_size = {}, "", "8"
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: 
                    text += t
                    m = re.search(r'Base Size\s*[:\s]\s*(\w+[\-?]?)', t, re.I)
                    if m: base_size = m.group(1).upper()

            for p in pdf.pages:
                for tb in p.extract_tables():
                    if not tb or len(tb) < 2: continue
                    h_idx = -1
                    for i, row in enumerate(tb[:10]):
                        row_up = [str(x).strip().upper() for x in row if x]
                        if any(base_size == x or (base_size in x and len(x) < 5) for x in row_up):
                            h_idx, header = i, [str(x).strip().upper() for x in row]
                            break
                    if h_idx == -1: continue
                    
                    base_idx = -1
                    for idx, h_val in enumerate(header):
                        if base_size == h_val or (base_size in h_val and len(h_val) < 5): base_idx = idx
                    
                    if base_idx != -1:
                        for r in tb[h_idx + 1:]:
                            if not r or len(r) <= base_idx: continue
                            desc = " ".join([str(x or "") for x in r[:base_idx]]).strip().upper()
                            val = parse_val(r[base_idx])
                            if val > 0.1 and len(desc) > 5:
                                specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:100]] = round(float(val), 3)
                            
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": "QUẦN/VÁY"}
    except: return None

# ================= SIDEBAR: NẠP KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    db_res = supabase.table("ai_data").select("*").execute()
    all_samples = db_res.data
    st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    
    # Nút chọn ngẫu nhiên
    if st.button("🎲 CHỌN MẪU NGẪU NHIÊN") and all_samples:
        st.session_state.random_sample = random.choice(all_samples)
        st.success(f"Đã chọn mã: {st.session_state.random_sample['file_name']}")

    st.divider()
    files = st.file_uploader("Nạp dữ liệu mới", accept_multiple_files=True, type=['pdf', 'xlsx'])
    # ... (Giữ nguyên logic upload cũ của bạn ở đây) ...

# ================= MAIN: SO SÁNH =================
st.title("👔 AI Fashion Pro - Đối Chiếu Thông Số")

test_file = st.file_uploader("Tải PDF Test", type="pdf")
target_sample = st.session_state.get('random_sample')

if test_file or target_sample:
    if test_file:
        with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
        data_test = get_data("test.pdf")
    else:
        # Nếu chưa có file test nhưng đã bấm ngẫu nhiên, ta mô phỏng lấy dữ liệu chính nó
        data_test = {"spec": target_sample['spec_json'], "img": None}

    if data_test and target_sample:
        st.subheader(f"🔍 So sánh với Mã Hàng: {target_sample['file_name']}")
        
        # Logic so sánh thông số
        rows = []
        db_specs = target_sample['spec_json']
        test_specs = data_test['spec']
        
        for k_test, v_test in test_specs.items():
            # Tìm thông số tương ứng trong DB bằng SequenceMatcher
            best_match = None
            highest_ratio = 0
            for k_db, v_db in db_specs.items():
                ratio = SequenceMatcher(None, k_test, k_db).ratio()
                if ratio > highest_ratio:
                    highest_ratio, best_match = ratio, k_db
            
            v_db = db_specs.get(best_match, 0) if highest_ratio > 0.6 else "N/A"
            diff = round(v_test - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
            status = "✅ OK" if diff == 0 else "❌ SAI" if diff != "N/A" else "❓ MỚI"
            
            rows.append({
                "Thông số PDF Test": k_test,
                "Số đo Test": v_test,
                "Thông số trong Kho": best_match if highest_ratio > 0.6 else "Không tìm thấy",
                "Số đo Kho": v_db,
                "Chênh lệch": diff,
                "Trạng thái": status
            })
        
        df_compare = pd.DataFrame(rows)
        
        # Hiển thị bảng so sánh
        st.dataframe(df_compare.style.applymap(
            lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''),
            subset=['Trạng thái']
        ), use_container_width=True, height=500)

        # Nút xuất file Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_compare.to_excel(writer, index=False, sheet_name='SoSanh')
        
        st.download_button(
            label="📥 XUẤT FILE EXCEL SO SÁNH",
            data=output.getvalue(),
            file_name=f"SoSanh_{target_sample['file_name']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
