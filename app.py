import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
import base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# CẤU HÌNH (Thay URL và KEY của bạn)
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro", page_icon="👔")

# --- HÀM HỖ TRỢ AI & ĐỌC PDF ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

def get_data(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode()
        
        specs = {}
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                tables = p.extract_tables()
                for table in tables:
                    for r in table:
                        if r and len(r) >= 2:
                            val_str = str(r[-1]).replace(',', '.')
                            nums = re.findall(r'\d+\.?\d*', val_str)
                            pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                            if nums and len(pom) > 3: specs[pom] = float(nums[0])
        return {"img_bytes": img_bytes, "img_b64": img_base64, "spec": specs}
    except: return None

# --- SIDEBAR: QUẢN TRỊ KHO ---
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    res = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res.count if res.count else 0)
    
    up_bulk = st.file_uploader("Nạp file mẫu PDF", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in up_bulk:
            with open("temp.pdf", "wb") as tmp: tmp.write(f.getbuffer())
            d = get_data("temp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v, "spec_json": d['spec'], "img_base64": d['img_b64']}).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- GIAO DIỆN CHÍNH ---
st.title("👔 AI Fashion - So sánh & Đối chiếu thông số")
up_test = st.file_uploader("📥 Tải file cần kiểm tra", type="pdf")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        # Tìm kiếm trong kho
        db = supabase.table("ai_data").select("*").execute()
        
        if not db.data:
            st.warning("Kho đang trống, không có gì để so sánh!")
            st.image(target['img_bytes'], caption="Ảnh file vừa tải lên", width=500)
        else:
            # Tính AI Similarity
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            sims = []
            for i in db.data:
                s = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                sims.append({"name": i['file_name'], "sim": s, "spec": i['spec_json'], "img": i.get('img_base64', '')})
            
            best = sorted(sims, key=lambda x: x['sim'], reverse=True)[0]

            # HIỂN THỊ 2 ẢNH SONG SONG
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🔍 Mẫu đang soi")
                st.image(target['img_bytes'], use_container_width=True)
            with col_b:
                st.subheader(f"✅ Mẫu khớp nhất: {best['name']} ({best['sim']:.1f}%)")
                if best['img']:
                    st.image(base64.b64decode(best['img']), use_container_width=True)
                else:
                    st.info("Mẫu này trong kho chưa có ảnh hiển thị.")

            # BẢNG SO SÁNH CHI TIẾT (4 CỘT)
            st.markdown("### 📊 Bảng đối chiếu thông số & Chênh lệch")
            diff_list = []
            all_poms = sorted(list(set(target['spec'].keys()) | set(best['spec'].keys())))
            
            for p in all_poms:
                v_new = target['spec'].get(p, 0)
                v_old = best['spec'].get(p, 0)
                diff = round(v_new - v_old, 3)
                diff_list.append({
                    "Thông số (POM)": p,
                    "Mẫu Mới": v_new,
                    "Mẫu Kho": v_old,
                    "Chênh lệch (+/-)": diff
                })
            
            df = pd.DataFrame(diff_list)
            st.dataframe(df, use_container_width=True) # Hiện bảng đầy đủ cột

            # NÚT XUẤT EXCEL
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                df.to_excel(wr, index=False, sheet_name='SoSanh')
            st.download_button(
                label="📥 TẢI BÁO CÁO EXCEL CHÊNH LỆCH",
                data=out.getvalue(),
                file_name=f"So_sanh_{up_test.name}.xlsx",
                mime="application/vnd.ms-excel"
            )
