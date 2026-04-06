import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, time, datetime
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

st.set_page_config(layout="wide", page_title="AI FASHION PRO V18.5 - POM MASS CHECKER", page_icon="🛡️")

# ================= HỆ THỐNG AI VISION (PHÂN LOẠI HÌNH ẢNH) =================
@st.cache_resource
def load_vision_ai():
    # Sử dụng ResNet50 - "Mắt thần" để soi dáng áo/quần
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

def get_vector(img_bytes):
    """Chuyển ảnh thành dãy số (Vector) để so sánh độ giống nhau"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

# ================= HỆ THỐNG TRÍCH XUẤT POM (THÔNG SỐ ĐO) =================
def extract_full_techpack(pdf_file):
    """Trích xuất sạch POM, bỏ qua BOM rác, lấy ảnh chất lượng cao"""
    specs, text = {}, ""
    try:
        pdf_bytes = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""; text += txt
                for tb in page.extract_tables():
                    if len(tb) < 2: continue
                    for row in tb:
                        # Lọc POM: Phải có tên điểm đo và có giá trị số
                        row_clean = [str(x).strip() for x in row if x]
                        if len(row_clean) >= 2:
                            key = row_clean[0].upper()
                            val = row_clean[-1]
                            # Loại bỏ các dòng tiêu đề rác hoặc BOM
                            if len(key) > 5 and any(c.isdigit() for c in val):
                                specs[key] = val
        
        # Lấy ảnh đại diện (Trang 1) để AI so sánh hình dáng
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: NẠP KHO HÀNG LOẠT =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU GỐC")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    
    # Cho phép chọn nhiều file PDF cùng lúc để nạp kho
    up_pdfs = st.file_uploader("Nạp PDF gốc (Chọn nhiều file)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 NẠP TẤT CẢ VÀO KHO"):
        progress_bar = st.progress(0)
        for i, f in enumerate(up_pdfs):
            with st.spinner(f"Đang xử lý: {f.name}"):
                d = extract_full_techpack(f)
                if d:
                    ma = f.name.replace(".pdf", "")
                    vec = get_vector(d['img'])
                    try:
                        # Lưu ảnh lên Storage
                        supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                        u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                        # Lưu thông số POM và Vector vào Database
                        supabase.table("ai_data").upsert({
                            "file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t
                        }).execute()
                    except Exception as e: st.error(f"Lỗi mã {ma}: {e}")
            progress_bar.progress((i + 1) / len(up_pdfs))
        st.success("✅ Đã nạp xong hàng loạt!"); st.rerun()

# ================= MAIN UI: SO SÁNH & ĐỒNG NHẤT HÀNG LOẠT =================
st.title("🛡️ AI FASHION POM - SIÊU SO SÁNH HÀNG LOẠT")
st.info("💡 Bạn chỉ cần tải các file cần kiểm tra lên. AI sẽ tự động bốc mã đúng nhất từ kho ra để đối chiếu POM.")

test_files = st.file_uploader("1. Tải các file PDF cần kiểm tra (Hàng loạt)", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        with st.expander(f"🔍 ĐANG KIỂM TRA: {t_file.name}", expanded=True):
            data_test = extract_full_techpack(t_file)
            
            if data_test:
                v_test = get_vector(data_test['img'])
                
                # --- TRỌNG TÂM: AI TÌM MÃ GẦN GIỐNG NHẤT (ƯU TIÊN HÌNH ẢNH) ---
                best_score, best_match = 0, None
                if samples and v_test:
                    for s in samples:
                        # Tính độ tương đồng Vector ảnh (Vision AI)
                        score = cosine_similarity([v_test], [s['vector']])[0][0]
                        if score > best_score:
                            best_score, best_match = score, s
                
                if best_match:
                    # Hiển thị 2 ảnh cạnh nhau để kiểm tra hình dáng
                    st.success(f"🤖 ĐÃ TÌM THẤY MÃ TƯƠNG ĐỒNG: **{best_match['file_name']}** (Độ giống hình ảnh: {best_score:.1%})")
                    
                    c1, c2 = st.columns(2)
                    with c1: st.image(data_test['img'], caption="BẢN VẼ FILE TEST", use_container_width=True)
                    with c2: st.image(best_match['img_url'], caption=f"MẪU GỐC TRONG KHO: {best_match['file_name']}", use_container_width=True)
                    
                    # --- SO SÁNH CHI TIẾT POM (ĐỒNG NHẤT THÔNG SỐ) ---
                    st.write("### 📏 BẢNG ĐỐI CHIẾU THÔNG SỐ POM")
                    
                    pom_test = data_test['spec']
                    pom_goc = best_match.get('spec_json', {})
                    
                    results = []
                    # Lấy tất cả các điểm đo từ cả 2 file để không bỏ sót hạng mục nào
                    all_keys = sorted(set(list(pom_test.keys()) + list(pom_goc.keys())))
                    
                    for k in all_keys:
                        v_t = pom_test.get(k, "Thiếu")
                        v_g = pom_goc.get(k, "Thiếu")
                        
                        # So sánh giá trị (Bỏ qua khoảng trắng và viết hoa)
                        status = "✅ KHỚP" if str(v_t).strip() == str(v_g).strip() else "❌ LỆCH"
                        
                        # Chỉ hiển thị những hạng mục có dữ liệu thực tế
                        if v_t != "Thiếu" or v_g != "Thiếu":
                            results.append({
                                "Hạng mục POM": k,
                                "File Test": v_t,
                                "Mẫu Gốc": v_g,
                                "Kết quả": status
                            })
                    
                    df_res = pd.DataFrame(results)
                    
                    # Áp dụng màu sắc cho bảng
                    def color_status(val):
                        color = 'red' if val == "❌ LỆCH" else 'green'
                        return f'color: {color}; font-weight: bold'
                    
                    st.table(df_res.style.applymap(color_status, subset=['Kết quả']))
                    
                    # Nút xuất file báo cáo riêng cho từng mã
                    csv = df_res.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(f"📥 Tải báo cáo: {t_file.name}.csv", csv, f"Report_{t_file.name}.csv", "text/csv")
                else:
                    st.error(f"❌ Không tìm thấy mẫu nào tương đồng trong kho cho file {t_file.name}")
