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
    st.error(f"❌ Kết nối Database thất bại: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V30.0", page_icon="🛡️")

# ================= 1. HỆ THỐNG AI VISION (SOI DÁNG SẢN PHẨM) =================
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

# ================= 2. HỆ THỐNG SOI CHI TIẾT (POCKETS, WAIST, SLEEVES) =================
def deep_detail_inspection(text):
    """
    AI Soi kỹ chi tiết thiết kế từ văn bản Techpack:
    Túi (Cargo, Slant, Patch, Welt), Lưng, Ống, Tay...
    """
    t = str(text).upper()
    details = {}
    
    # Soi Túi trước (Pockets)
    pockets = []
    if 'CARGO' in t or 'BOX POCKET' in t: pockets.append("📦 Túi Hộp (Cargo)")
    if 'SLANT' in t or 'SIDE POCKET' in t: pockets.append("📐 Túi Xéo (Slant)")
    if 'PATCH POCKET' in t or 'TUI DAP' in t: pockets.append("🎨 Túi Đắp (Patch)")
    if 'WELT' in t or 'BONE' in t or 'TUI MO' in t: pockets.append("✂️ Túi Mổ (Welt/Bone)")
    if 'SCOOP' in t or 'HAM ECH' in t: pockets.append("🐸 Túi Hàm Ếch (Scoop)")
    details['Túi'] = ", ".join(pockets) if pockets else "Không xác định"

    # Soi Lưng & Cạp (Waistband)
    if 'ELASTIC' in t: details['Lưng'] = "🧶 Lưng Thun (Elastic)"
    elif 'RIB' in t: details['Lưng'] = "🧶 Bo Rib"
    else: details['Lưng'] = "🧵 Lưng Vải Chính"

    # Soi các chi tiết khác
    if 'SKORT' in t: details['Loại'] = "👗 Quần Váy (Skort)"
    if 'ZIPPER' in t: details['Phụ liệu'] = "🔩 Có Dây Kéo"
    
    return details

# ================= 3. TRÍCH XUẤT SIÊU CẤP (FULL CRITERIA) =================
def extract_comprehensive_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    pom_keywords = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE', 'RISE', 'INSEAM', 'THIGH', 'LEG', 'OPENING']
    
    try:
        pdf_bytes = pdf_file.read()
        # Lấy ảnh đại diện trang 1
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).tobytes("png")
        doc.close()

        # Quét POM và Text để soi chi tiết
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                tables = page.extract_tables()
                for tb in tables:
                    for row in tb:
                        cells = [str(c).strip() for c in row if c and str(c).strip()]
                        if len(cells) < 2: continue
                        
                        key = cells.upper()
                        if any(k in key for k in pom_keywords):
                            val = ""
                            for c in reversed(cells[1:]):
                                if any(char.isdigit() for char in c):
                                    val = c; break
                            if val: specs[key] = val
        
        if img and len(specs) >= 2:
            details = deep_detail_inspection(raw_text)
            return {"spec": specs, "img": img, "details": details}
        return None
    except: return None

# ================= 4. SIDEBAR: QUẢN LÝ KHO ĐA TIÊU CHÍ =================
with st.sidebar:
    st.header("📦 KHO MẪU CHUẨN V30")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc (Bulk)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 NẠP VÀ PHÂN TÍCH CHI TIẾT"):
        for f in up_pdfs:
            d = extract_comprehensive_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                    supabase.table("ai_data").upsert({
                        "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                        "img_url": u_t, "details": d['details']
                    }).execute()
                    st.toast(f"✅ Đã nạp: {ma}")
                except: pass
        st.rerun()

# ================= 5. MAIN UI: SIÊU ĐỐI CHIẾU & PHÂN LOẠI =================
st.title("🛡️ AI FASHION AUDITOR - ĐỐI CHIẾU TOÀN DIỆN")

test_files = st.file_uploader("1. Tải Techpack cần kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_comprehensive_techpack(t_file)
        if not data_test:
            st.warning(f"🚫 File không đạt chuẩn POM/Ảnh: {t_file.name}"); continue

        with st.expander(f"🔍 ĐANG SOI CHI TIẾT: {t_file.name}", expanded=True):
            v_test = get_vector(data_test['img'])
            best_score, best_match = 0.0, None
            
            # AI TÌM MÃ GẦN GIỐNG NHẤT (Vision priority)
            if samples and v_test:
                for s in samples:
                    sim = float(cosine_similarity([v_test], [s['vector']]))
                    if sim > best_score: best_score, best_match = sim, s
            
            if best_match:
                st.success(f"🤖 ĐÃ KHỚP MÃ: **{best_match['file_name']}** (Độ giống dáng: {best_score:.1%})")
                
                # --- PHẦN 1: SOI CHI TIẾT THIẾT KẾ ---
                st.subheader("🕵️ 1. SOI CHI TIẾT THIẾT KẾ (DESIGN DETAILS)")
                det_test = data_test['details']
                det_goc = best_match.get('details', {})
                
                col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
                for category in ['Túi', 'Lưng', 'Loại', 'Phụ liệu']:
                    v1 = det_test.get(category, "N/A")
                    v2 = det_goc.get(category, "N/A")
                    match = "✅ ĐỒNG NHẤT" if v1 == v2 else "❌ KHÁC BIỆT"
                    with st.container():
                        st.write(f"**{category}:** {v1} vs {v2} -> {match}")

                # --- PHẦN 2: SO SÁNH HÌNH ẢNH ---
                st.subheader("🖼️ 2. ĐỐI CHIẾU HÌNH ẢNH TRỰC QUAN")
                c1, c2 = st.columns(2)
                with c1: st.image(data_test['img'], caption="BẢN VẼ FILE TEST", use_container_width=True)
                with c2: st.image(best_match['img_url'], caption=f"MẪU GỐC: {best_match['file_name']}", use_container_width=True)
                
                # --- PHẦN 3: ĐỐI CHIẾU THÔNG SỐ POM ---
                st.subheader("📏 3. BẢNG ĐỐI CHIẾU THÔNG SỐ KỸ THUẬT (POM)")
                res_comp = []
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                all_keys = sorted(set(list(p_t.keys()) + list(p_g.keys())))
                
                for k in all_keys:
                    v_t, v_g = p_t.get(k, "---"), p_g.get(k, "---")
                    status = "✅ KHỚP" if str(v_t).strip() == str(v_g).strip() else "❌ LỆCH"
                    if v_t != "---" or v_g != "---":
                        res_comp.append({"Hạng mục POM": k, "Test": v_t, "Gốc": v_g, "Kết quả": status})
                
                df_res = pd.DataFrame(res_comp)
                
                def style_results(val):
                    if val == "❌ LỆCH": return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'

                st.dataframe(df_res.style.map(style_results, subset=['Kết quả']), use_container_width=True)
                
                # XUẤT BÁO CÁO
                csv = df_res.to_csv(index=False).encode('utf-8-sig')
                st.download_button(f"📥 TẢI BÁO CÁO KIỂM ĐỊNH: {t_file.name}", csv, f"Auditor_Report_{t_file.name}.csv", "text/csv")
            else:
                st.error(f"❌ Không tìm thấy mẫu nào trong kho có dáng tương đồng với {t_file.name}")
