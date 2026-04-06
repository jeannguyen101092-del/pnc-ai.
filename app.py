import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"             
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Database!")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V16.2", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 1500

@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

# ================= HỆ THỐNG XỬ LÝ AI =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

def extract_pdf_ultimate(pdf_file):
    specs, text = {}, ""
    try:
        pdf_bytes = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""; text += txt
                for tb in page.extract_tables():
                    for row in tb:
                        if row and len(row) >= 2:
                            # Lọc lấy các dòng có số (thường là thông số)
                            key = str(row[0]).strip().upper()
                            val = str(row[1]).strip()
                            if len(key) > 3: specs[key] = val
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU V16.2")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    list_ma = [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU", ["-- Click chọn --"] + list_ma)
    if sel != "-- Click chọn --": 
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    up_pdf = st.file_uploader("Nạp PDF gốc vào kho", type=['pdf'], key=st.session_state.up_key)
    if up_pdf and st.button("🚀 NẠP VÀO KHO"):
        with st.spinner("Đang nạp..."):
            d = extract_pdf_ultimate(up_pdf)
            if d:
                ma = up_pdf.name.split('.')[0]
                vec = get_vector(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                    supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t}).execute()
                    st.toast(f"✅ Đã nạp mã {ma}")
                    st.session_state.up_key += 1; st.rerun()
                except Exception as e: st.error(f"Lỗi: {e}")

# ================= MAIN UI: SO SÁNH =================
st.title("🛡️ AI FASHION PRO - SO SÁNH & TÌM KIẾM")

test_pdf = st.file_uploader("1. Tải PDF cần kiểm tra", type="pdf")

if test_pdf:
    data_test = extract_pdf_ultimate(test_pdf)
    if data_test:
        col1, col2 = st.columns(2)
        with col1:
            st.image(data_test['img'], caption="HÌNH ẢNH FILE TEST", use_container_width=True)
            
        # Nút Tìm kiếm tương đồng
        if st.button("🤖 AI: TỰ ĐỘNG TÌM MÃ GIỐNG NHẤT TRONG KHO"):
            v_test = get_vector(data_test['img'])
            if samples and v_test:
                best_s, best_m = 0, None
                for s in samples:
                    score = cosine_similarity([v_test], [s['vector']])[0][0]
                    if score > best_s: best_s, best_m = score, s
                if best_m:
                    st.session_state.target = best_m
                    st.success(f"✅ Đã tìm thấy mã {best_m['file_name']} giống {best_s:.1%}")
                    st.rerun()

        # Hiển thị mã đối chiếu và So sánh
        target = st.session_state.target
        if target:
            with col2:
                st.image(target['img_url'], caption=f"KHO GỐC: {target['file_name']}", use_container_width=True)
            
            st.divider()
            st.subheader(f"📊 BẢNG SO SÁNH THÔNG SỐ: TEST vs {target['file_name']}")
            
            # Logic so sánh bảng thông số
            compare_list = []
            test_specs = data_test['spec']
            target_specs = target.get('spec_json', {})
            
            all_keys = set(list(test_specs.keys()) + list(target_specs.keys()))
            for k in sorted(all_keys):
                v_test = test_specs.get(k, "N/A")
                v_goc = target_specs.get(k, "N/A")
                status = "✅ Khớp" if str(v_test) == str(v_goc) else "❌ Lệch"
                if v_test != "N/A" or v_goc != "N/A":
                    compare_list.append({"Hạng mục": k, "Giá trị Test": v_test, "Giá trị Gốc": v_goc, "Kết quả": status})
            
            df_compare = pd.DataFrame(compare_list)
            st.table(df_compare)
            
            # NÚT XUẤT FILE BÁO CÁO
            csv = df_compare.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 TẢI BÁO CÁO SO SÁNH (CSV)",
                data=csv,
                file_name=f"SoSanh_{target['file_name']}.csv",
                mime='text/csv'
            )
        else:
            with col2:
                st.info("👈 Hãy chọn một mã ở Sidebar hoặc bấm nút AI để bắt đầu đối chiếu.")

# Kết thúc code
