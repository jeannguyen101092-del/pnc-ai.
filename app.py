import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, time
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
except Exception as e:
    st.error(f"❌ Lỗi kết nối Supabase: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI POM PRO V25.0", page_icon="🛡️")

# ================= HỆ THỐNG AI VISION (ƯU TIÊN NHẬN DIỆN DÁNG) =================
@st.cache_resource
def load_vision_ai():
    # Dùng ResNet50 để "nhìn" và phân loại dáng sản phẩm (áo, quần, váy...)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

# ================= SIÊU TRÍCH XUẤT POM (PHÂN TÍCH SÂU TỪNG Ô) =================
def extract_valid_techpack(pdf_file):
    """
    Tự động lọc file: Phải có ảnh dáng và có thông số POM.
    Nếu file rác hoặc không lấy được thông số -> BỎ QUA.
    """
    specs, img = {}, None
    try:
        pdf_bytes = pdf_file.read()
        # 1. Trích xuất ảnh trang đầu (Dùng làm mẫu so sánh hình ảnh)
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
            doc.close()
        except: img = None

        # 2. Thuật toán trích xuất POM 'Aggressive' (Lấy sạch không bỏ sót)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables: continue
                for tb in tables:
                    for row in tb:
                        # Làm sạch dòng: bỏ None, bỏ ô trống
                        row_clean = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                        if len(row_clean) < 2: continue
                        
                        # Tìm Key (Tên điểm đo) và Val (Thông số thực tế)
                        found_key, found_val = None, None
                        for i, cell in enumerate(row_clean):
                            # Tên POM thường nằm ở 2 cột đầu và không được là số thuần túy
                            if i < 2 and not found_key and len(cell) > 3 and not cell.replace('.','').isdigit():
                                found_key = cell.upper()
                            # Thông số thường chứa chữ số (ví dụ: 12, 1/2, 24.5)
                            if i > 0 and any(c.isdigit() for c in cell):
                                found_val = cell
                        
                        if found_key and found_val:
                            specs[found_key] = found_val
        
        # ĐIỀU KIỆN ĐỂ NẠP KHO: Có hình và tối thiểu 3 dòng thông số
        if img and len(specs) >= 3:
            return {"spec": specs, "img": img}
        return None
    except: return None

# ================= SIDEBAR: NẠP KHO HÀNG LOẠT =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU GỐC")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc (Chọn nhiều file)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 NẠP VÀO KHO"):
        for f in up_pdfs:
            d = extract_valid_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                    supabase.table("ai_data").upsert({
                        "file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t
                    }).execute()
                    st.toast(f"✅ Đã nạp: {ma}")
                except: pass
            else: st.error(f"⚠️ File không hợp lệ/Thiếu POM: {f.name}")
        st.rerun()

# ================= MAIN UI: TỰ ĐỘNG ĐỐI CHIẾU =================
st.title("🛡️ AI POM PRO V25.0 - SIÊU ĐỐI CHIẾU")
st.info("💡 Hệ thống tự động tìm mã giống nhất dựa trên hình ảnh, sau đó đối chiếu thông số POM.")

test_files = st.file_uploader("1. Tải các file cần kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_valid_techpack(t_file)
        if not data_test:
            st.warning(f"🚫 File không tách được thông số hoặc ảnh: {t_file.name}")
            continue

        with st.expander(f"🔍 ĐANG PHÂN TÍCH: {t_file.name}", expanded=True):
            v_test = get_vector(data_test['img'])
            
            # --- TÌM MÃ GIỐNG NHẤT (FIX LỖI TYPEERROR) ---
            best_score, best_match = 0.0, None
            if samples and v_test:
                for s in samples:
                    # Lấy kết quả chính xác từ ma trận similarity
                    sim_matrix = cosine_similarity([v_test], [s['vector']])
                    score = float(sim_matrix[0][0])
                    if score > best_score:
                        best_score, best_match = score, s
            
            if best_match:
                st.success(f"🤖 AI nhận diện mã: **{best_match['file_name']}** (Giống hình ảnh: {best_score:.1%})")
                c1, c2 = st.columns(2)
                with c1: st.image(data_test['img'], caption="BẢN VẼ TEST", use_container_width=True)
                with c2: st.image(best_match['img_url'], caption=f"MẪU KHO: {best_match['file_name']}", use_container_width=True)
                
                st.write("### 📐 BẢNG ĐỐI CHIẾU THÔNG SỐ POM")
                res_comp = []
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                all_keys = sorted(set(list(p_t.keys()) + list(p_g.keys())))
                
                for k in all_keys:
                    v_t, v_g = p_t.get(k, "---"), p_g.get(k, "---")
                    status = "✅ KHỚP" if str(v_t).strip() == str(v_g).strip() else "❌ LỆCH"
                    if v_t != "---" or v_g != "---":
                        res_comp.append({"Hạng mục POM": k, "Test": v_t, "Gốc": v_g, "Kết quả": status})
                
                df_res = pd.DataFrame(res_comp)

                # HIỂN THỊ BẢNG MÀU SẮC (SỬA LỖI ATTRIBUTEERROR)
                def color_rows(val):
                    color = 'background-color: #f8d7da; color: #721c24' if val == "❌ LỆCH" else 'background-color: #d4edda; color: #155724'
                    return color

                st.dataframe(
                    df_res.style.applymap(color_rows, subset=['Kết quả']),
                    use_container_width=True
                )
                
                st.download_button(f"📥 Tải báo cáo: {t_file.name}.csv", df_res.to_csv(index=False).encode('utf-8-sig'), f"Report_{t_file.name}.csv", "text/csv")
            else:
                st.error(f"❌ Không tìm thấy mẫu tương đồng trong kho cho {t_file.name}")

# NHẮC LẠI SQL: ALTER TABLE ai_data DISABLE ROW LEVEL SECURITY;
