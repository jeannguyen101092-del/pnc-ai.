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
URL ="https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V4.7", page_icon="👔")

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
        img_b64 = base64.b64encode(img_bytes).decode()
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
        return {"img_bytes": img_bytes, "img_b64": img_b64, "spec": specs}
    except: return None

# --- SIDEBAR: QUẢN TRỊ KHO ---
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng mẫu trong kho", res_db.count if res_db.count else 0)
    except: st.error("Lỗi kết nối!")
    
    up_bulk = st.file_uploader("Nạp mẫu mới vào kho", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in up_bulk:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v, "spec_json": d['spec'], "img_base64": d['img_b64']}).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- GIAO DIỆN CHÍNH ---
st.title("👔 AI Fashion - So sánh & Tìm kiếm thông minh")
up_test = st.file_uploader("📥 Tải file cần kiểm tra (PDF)", type="pdf")

if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        db = supabase.table("ai_data").select("*").execute()
        if not db.data:
            st.warning("⚠️ Kho đang trống, vui lòng nạp mẫu vào bên trái!")
        else:
            # --- TÍNH TOÁN SO SÁNH AN TOÀN (CHỐNG LỖI ĐỎ) ---
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            sims = []
            for i in db.data:
                # KIỂM TRA CHẶN LỖI TYPEERROR: Chỉ so sánh nếu có vector hợp lệ
                if i.get('vector') is not None and isinstance(i['vector'], list) and len(i['vector']) > 0:
                    s = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    sims.append({"name": i['file_name'], "sim": s, "spec": i['spec_json'], "img": i['img_base64']})
            
            if sims:
                best = sorted(sims, key=lambda x: x['sim'], reverse=True)[0]
                col1, col2 = st.columns(2)
                with col1:
                    st.image(target['img_bytes'], caption="Mẫu mới", use_container_width=True)
                with col2:
                    st.image(base64.b64decode(best['img']), caption=f"Mẫu khớp: {best['name']}", use_container_width=True)

                # BẢNG SO SÁNH 4 CỘT & NÚT EXCEL
                diff_list = []
                poms = sorted(list(set(target['spec'].keys()) | set(best['spec'].keys())))
                for p in poms:
                    v_n, v_o = target['spec'].get(p, 0), best['spec'].get(p, 0)
                    diff_list.append({"Thông số": p, "Mẫu Mới": v_n, "Mẫu Kho": v_old if (v_old := v_o) else 0, "Chênh lệch": round(v_n - v_o, 2)})
                
                df = pd.DataFrame(diff_list)
                st.table(df) # Hiện bảng đủ 4 cột
                
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    df.to_excel(wr, index=False)
                st.download_button("📥 TẢI EXCEL SO SÁNH", out.getvalue(), "Ket_qua.xlsx")
            else:
                st.error("Dữ liệu trong kho không hợp lệ. Hãy xóa mẫu cũ và nạp lại!")
