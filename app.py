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
    st.error(f"❌ Lỗi kết nối Database: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI POM CHECKER V22.0", page_icon="🛡️")

# ================= HỆ THỐNG AI VISION (ƯU TIÊN HÌNH ẢNH) =================
@st.cache_resource
def load_vision_ai():
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

# ================= BỘ LỌC TRÍCH XUẤT (LOẠI BỎ FILE LỖI) =================
def extract_valid_techpack(pdf_file):
    """Chỉ lấy file có ĐỦ Ảnh và ĐỦ Thông số POM. Không đạt -> BỎ QUA."""
    specs, img, text = {}, None, ""
    try:
        pdf_bytes = pdf_file.read()
        # 1. Kiểm tra Ảnh (Trang 1)
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
            img = pix.tobytes("png")
            doc.close()
        except: img = None

        # 2. Kiểm tra Thông số POM (Point of Measure)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "")
                for tb in page.extract_tables():
                    for row in tb:
                        row_c = [str(x).strip() for x in row if x]
                        if len(row_c) >= 2:
                            key = row_c[0].upper()
                            val = row_c[-1]
                            # Lọc POM: Có chữ cái ở key và có số đo thực tế
                            if len(key) > 4 and any(c.isdigit() for c in val):
                                specs[key] = val
        
        # ĐIỀU KIỆN SỐNG CÒN: Phải có ảnh VÀ có ít nhất 3 dòng thông số POM
        if img and len(specs) >= 3:
            return {"spec": specs, "img": img, "text": text}
        return None
    except: return None

# ================= SIDEBAR: NẠP KHO SẠCH =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU GỐC")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc (Hàng loạt)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 LỌC & NẠP VÀO KHO"):
        for f in up_pdfs:
            d = extract_valid_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                    supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t}).execute()
                    st.toast(f"✅ Đã nạp: {ma}")
                except: pass
            else: st.error(f"⚠️ File lỗi (Thiếu POM/Ảnh) -> Đã bỏ qua: {f.name}")
        st.rerun()

# ================= MAIN UI: AUTO-MATCH & SIÊU SO SÁNH =================
st.title("🛡️ AI POM CHECKER V22.0 - SIÊU SO SÁNH")
st.info("💡 Hệ thống tự động bỏ qua file lỗi. Ưu tiên khớp hình dáng rồi mới đối chiếu thông số.")

test_files = st.file_uploader("1. Tải các file cần kiểm tra (Hàng loạt)", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_valid_techpack(t_file)
        if not data_test:
            st.warning(f"🚫 File không hợp lệ (Không có POM/Ảnh): {t_file.name}")
            continue

        with st.expander(f"🔍 ĐANG ĐỐI CHIẾU: {t_file.name}", expanded=True):
            v_test = get_vector(data_test['img'])
            
            # --- TÌM MÃ GIỐNG NHẤT (FIX LỖI TYPE ERROR TẠI ĐÂY) ---
            best_s, best_m = 0.0, None
            if samples and v_test:
                for s in samples:
                    # Lấy giá trị float từ mảng kết quả của cosine_similarity
                    score = float(cosine_similarity([v_test], [s['vector']])[0][0])
                    if score > best_s:
                        best_s, best_m = score, s
            
            if best_m:
                st.success(f"🤖 Khớp với: **{best_m['file_name']}** (Độ giống dáng: {best_s:.1%})")
                c1, c2 = st.columns(2)
                with c1: st.image(data_test['img'], caption="BẢN VẼ FILE TEST", use_container_width=True)
                with c2: st.image(best_m['img_url'], caption=f"MẪU KHO: {best_m['file_name']}", use_container_width=True)
                
                st.write("### 📐 BẢNG ĐỐI CHIẾU THÔNG SỐ POM")
                res = []
                p_t, p_g = data_test['spec'], best_m.get('spec_json', {})
                all_k = sorted(set(list(p_t.keys()) + list(p_g.keys())))
                
                for k in all_k:
                    v_t, v_g = p_t.get(k, "---"), p_g.get(k, "---")
                    status = "✅ KHỚP" if str(v_t).strip() == str(v_g).strip() else "❌ LỆCH"
                    if v_t != "---" or v_g != "---":
                        res.append({"Hạng mục POM": k, "Test": v_t, "Gốc": v_g, "Kết quả": status})
                
                df_res = pd.DataFrame(res)

                # HIỂN THỊ BẢNG MÀU SẮC
                def style_result(val):
                    if val == "❌ LỆCH": return 'background-color: #f8d7da; color: #721c24'
                    return 'background-color: #d4edda; color: #155724'

                st.dataframe(
                    df_res.style.applymap(style_result, subset=['Kết quả']),
                    use_container_width=True
                )
                
                csv = df_res.to_csv(index=False).encode('utf-8-sig')
                st.download_button(f"📥 Tải báo cáo: {t_file.name}.csv", csv, f"Report_{t_file.name}.csv", "text/csv")
            else:
                st.error(f"❌ Không tìm thấy mã tương đồng cho {t_file.name}")
