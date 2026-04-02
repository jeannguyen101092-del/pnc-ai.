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
# 1. CẤU HÌNH (Thay URL và KEY của bạn)
# ==========================================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro", page_icon="👔")

# --- HÀM HỖ TRỢ AI ---
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
        res = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng mẫu trong kho", res.count if res.count else 0)
    except: st.error("Chưa kết nối được Database!")
    
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
        db = supabase.table("ai_data").select("*").execute()
        
        if not db.data:
            st.warning("Kho đang trống!")
        else:
            # TÍNH TOÁN SO SÁNH AN TOÀN (Sửa lỗi TypeError)
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            sims = []
            for i in db.data:
                # Kiểm tra vector có hợp lệ không trước khi so sánh
                if i.get('vector') and len(i['vector']) > 0:
                    s = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    sims.append({"name": i['file_name'], "sim": s, "spec": i['spec_json'], "img": i.get('img_base64', '')})
            
            if not sims:
                st.error("Dữ liệu trong kho bị lỗi Vector. Vui lòng nạp lại mẫu!")
            else:
                best = sorted(sims, key=lambda x: x['sim'], reverse=True)[0]

                # HIỂN THỊ 2 ẢNH
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🔍 Mẫu đang soi")
                    st.image(target['img_bytes'], use_container_width=True)
                with c2:
                    st.subheader(f"✅ Khớp nhất: {best['name']} ({best['sim']:.1f}%)")
                    if best['img']:
                        st.image(base64.b64decode(best['img']), use_container_width=True)

                # BẢNG SO SÁNH (4 CỘT)
                st.markdown("### 📊 Bảng đối chiếu thông số & Chênh lệch")
                diff_list = []
                all_poms = sorted(list(set(target['spec'].keys()) | set(best['spec'].keys())))
                
                for p in all_poms:
                    v_new, v_old = target['spec'].get(p, 0), best['spec'].get(p, 0)
                    diff_list.append({
                        "Thông số (POM)": p,
                        "Mẫu Mới": v_new,
                        "Mẫu Kho": v_old,
                        "Chênh lệch": round(v_new - v_old, 2)
                    })
                
                df = pd.DataFrame(diff_list)
                st.table(df) # Hiện bảng 4 cột rõ ràng

                # NÚT EXCEL
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    df.to_excel(wr, index=False)
                st.download_button("📥 TẢI EXCEL SO SÁNH", out.getvalue(), f"So_sanh_{up_test.name}.xlsx")
