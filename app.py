import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. KẾT NỐI SUPABASE
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co" 
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI SMART SPEC PRO V4.2", page_icon="🔍")

# Khởi tạo bộ nhớ tạm
if 'sel_code' not in st.session_state: st.session_state.sel_code = None

@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

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
                            val_str = str(r[-1]).replace(',', '.')
                            nums = re.findall(r'\d+\.?\d*', val_str)
                            pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                            if nums and len(pom) > 3: specs[pom] = float(nums[0])
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
        cat = "QUẦN" if "PANT" in all_texts.upper() else "ÁO"
        return {"spec": specs, "img_b64": img_b64, "img_bytes": pix.tobytes("png"), "cat": cat}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_db.count if res_db.count else 0)
    
    up_bulk = st.file_uploader("Nạp mẫu mới vào kho", accept_multiple_files=True)
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

# --- GIAO DIỆN CHÍNH ---
st.title("👔 AI Fashion - So sánh & Tìm kiếm thông minh")

up_test = st.file_uploader("📥 Tải file cần kiểm tra (PDF)", type="pdf")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        # Lấy danh sách tên file trong kho để Tìm kiếm
        all_items = supabase.table("ai_data").select("file_name, spec_json, img_base64, vector").execute()
        file_list = [i['file_name'] for i in all_items.data]

        st.divider()
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        
        with col_ctrl1:
            mode = st.radio("🎯 Chế độ so sánh:", ["Tự động (AI)", "Tìm mã thủ công"], horizontal=True)
        
        with col_ctrl2:
            if mode == "Tìm mã thủ công":
                st.session_state.sel_code = st.selectbox("🔍 Gõ tên mã hàng để tìm:", file_list)
            else:
                st.info("💡 Hệ thống đang tự động tìm mã khớp nhất...")

        # XỬ LÝ LẤY DỮ LIỆU ĐỐI CHIẾU
        best_match = None
        
        if mode == "Tự động (AI)":
            # Tính AI Similarity
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            sims = []
            for i in all_items.data:
                s = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                sims.append({"name": i['file_name'], "sim": s, "spec": i['spec_json'], "img": i['img_base64']})
            
            best_match = sorted(sims, key=lambda x: x['sim'], reverse=True)[0]
        else:
            # Lấy dữ liệu từ mã chọn thủ công
            selected_item = next(i for i in all_items.data if i['file_name'] == st.session_state.sel_code)
            best_match = {"name": selected_item['file_name'], "sim": 0, "spec": selected_item['spec_json'], "img": selected_item['img_base64']}

        # HIỂN THỊ KẾT QUẢ
        if best_match:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔍 Đang soi: " + up_test.name)
                st.image(target['img_bytes'], use_container_width=True)
            with c2:
                st.subheader(f"✅ Đối chiếu với: {best_match['name']}")
                st.image(base64.b64decode(best_match['img']), use_container_width=True)

            # BẢNG SO SÁNH 4 CỘT
            st.markdown("### 📊 Bảng đối chiếu thông số & Chênh lệch")
            diff_list = []
            poms = sorted(list(set(target['spec'].keys()) | set(best_match['spec'].keys())))
            for p in poms:
                v_new, v_old = target['spec'].get(p, 0), best_match['spec'].get(p, 0)
                diff = round(v_new - v_old, 2)
                diff_list.append({"Thông số (POM)": p, "Mẫu Mới": v_new, "Mẫu Kho": v_old, "Chênh lệch": diff})
            
            df = pd.DataFrame(diff_list)
            st.table(df)

            # NÚT XUẤT EXCEL
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                df.to_excel(wr, index=False)
            st.download_button("📥 TẢI EXCEL SO SÁNH", out.getvalue(), f"So_sanh_{best_match['name']}.xlsx")
