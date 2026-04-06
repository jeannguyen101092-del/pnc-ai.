import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL và KEY thực tế) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"                
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Database!")

st.set_page_config(layout="wide", page_title="AI FASHION POM CHECKER", page_icon="📏")

# Khởi tạo AI ResNet50 để lấy Vector ảnh
@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

def extract_pom_only(pdf_file):
    """Hàm chuyên trích xuất thông số POM, bỏ qua các dữ liệu khác"""
    specs, text = {}, ""
    try:
        pdf_bytes = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""; text += txt
                for tb in page.extract_tables():
                    for row in tb:
                        # Logic lọc POM: Thường dòng POM sẽ có ít nhất 2 cột và cột cuối là số/thông số
                        if row and len(row) >= 2:
                            key = str(row[0]).strip().upper()
                            val = str(row[-1]).strip() # Lấy giá trị ở cột cuối
                            # Chỉ lấy nếu Key dài và có nội dung kỹ thuật (ví dụ: Waist, Hip, Inseam...)
                            if len(key) > 5 and any(char.isdigit() for char in val):
                                specs[key] = val
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU POM")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("MẪU TRONG KHO", len(samples))
    
    list_ma = ["-- Tự động tìm mã --"] + [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU", list_ma)
    
    # Cập nhật target dựa trên lựa chọn hoặc để None để Auto
    if sel != "-- Tự động tìm mã --":
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)
    else:
        st.session_state.target = None

    st.divider()
    up_pdf = st.file_uploader("Nạp mẫu mới vào kho", type=['pdf'])
    if up_pdf and st.button("🚀 NẠP MẪU"):
        with st.spinner("Đang nạp..."):
            d = extract_pom_only(up_pdf)
            if d:
                ma = up_pdf.name.split('.')[0]
                vec = get_vector(d['img'])
                supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t}).execute()
                st.toast("✅ Đã nạp thành công!"); st.rerun()

# ================= MAIN UI =================
st.title("🛡️ AI FASHION POM - AUTO CHECKER")

test_file = st.file_uploader("1. Tải PDF cần kiểm tra POM", type="pdf")

if test_file:
    data_test = extract_pom_only(test_pdf)
    if data_test:
        v_test = get_vector(data_test['img'])
        
        # AUTO SEARCH: Nếu không chọn mã thủ công, AI tự tìm mã giống nhất
        if st.session_state.target is None and samples:
            best_s, best_m = 0, None
            for s in samples:
                score = cosine_similarity([v_test], [s['vector']])[0][0]
                if score > best_s: best_s, best_m = score, s
            st.session_state.target = best_m
            if best_m:
                st.success(f"🤖 AI Tự động nhận diện mã: **{best_m['file_name']}** (Độ giống: {best_s:.1%})")

        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(data_test['img'], caption="BẢN VẼ FILE TEST", use_container_width=True)
        
        if target:
            with col2:
                st.image(target['img_url'], caption=f"MẪU ĐỐI CHIẾU: {target['file_name']}", use_container_width=True)
            
            st.divider()
            st.subheader("📏 SO SÁNH THÔNG SỐ POM")
            
            # Tạo bảng so sánh
            compare_rows = []
            test_pom = data_test['spec']
            target_pom = target.get('spec_json', {})
            
            for k in sorted(test_pom.keys()):
                v1 = test_pom[k]
                v2 = target_pom.get(k, "---")
                status = "✅ Khớp" if str(v1) == str(v2) else "❌ Lệch"
                compare_rows.append({"Hạng mục POM": k, "Giá trị Test": v1, "Kho Gốc": v2, "Kết quả": status})
            
            df_res = pd.DataFrame(compare_rows)
            st.table(df_res)
            
            # Nút xuất file báo cáo
            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 XUẤT BÁO CÁO POM (CSV)", csv, f"POM_Check_{target['file_name']}.csv", "text/csv")
        else:
            with col2: st.info("Chưa tìm thấy mã tương ứng trong kho để so sánh.")
