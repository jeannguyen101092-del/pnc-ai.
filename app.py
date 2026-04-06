import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (URL/KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"              
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Lỗi kết nối Supabase: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI POM PRO V28.0", page_icon="📏")

# ================= HỆ THỐNG AI VISION (MẮT THẦN) =================
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

# ================= BỘ LỌC POM SIÊU SẠCH (CHỈ LẤY THÔNG SỐ ĐO) =================
def extract_clean_pom(pdf_file):
    specs, img = {}, None
    # Từ khóa rác cần loại bỏ tuyệt đối
    trash = ['NOTE', 'MATERIAL', 'FABRIC', 'LABEL', 'COLOR', 'DESCRIPTION', 'ATTRIBUTE', 'REF', 'DESIGN']
    # Từ khóa vàng của POM
    pom_gold = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE', 'RISE', 'INSEAM', 'THIGH', 'LEG', 'OPENING']
    
    try:
        pdf_bytes = pdf_file.read()
        # 1. Trích xuất ảnh dáng sản phẩm
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()

        # 2. Quét POM chuyên sâu (Deep Scan)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    for row in tb:
                        cells = [str(c).strip() for c in row if c and str(c).strip()]
                        if len(cells) < 2: continue
                        
                        txt_row = " ".join(cells).upper()
                        # KIỂM TRA: Phải chứa từ khóa POM vàng VÀ không chứa từ khóa rác
                        is_pom = any(k in txt_row for k in pom_gold)
                        is_trash = any(k in txt_row for k in trash)
                        
                        if is_pom and not is_trash:
                            key = cells[0].upper() if not cells[0].replace('.','').isdigit() else cells[1].upper()
                            # Tìm ô cuối cùng có chứa số (thông số đo)
                            val = ""
                            for c in reversed(cells):
                                if any(char.isdigit() for char in c) and len(c) < 10:
                                    val = c; break
                            if len(key) > 3 and val:
                                specs[key] = val
        
        if img and len(specs) >= 2:
            return {"spec": specs, "img": img}
        return None
    except: return None

# ================= SIDEBAR: NẠP KHO HÀNG LOẠT =================
with st.sidebar:
    st.header("📦 KHO GỐC")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc (Bulk)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 LỌC & NẠP"):
        for f in up_pdfs:
            d = extract_clean_pom(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")
                    supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t}).execute()
                    st.toast(f"✅ Đã nạp: {ma}")
                except: pass
            else: st.error(f"⚠️ Bỏ qua file rác: {f.name}")
        st.rerun()

# ================= MAIN UI: AUTO SEARCH & SO SÁNH =================
st.title("🛡️ AI POM PRO V28.0 - SIÊU ĐỐI CHIẾU")

test_files = st.file_uploader("1. Tải các file cần kiểm tra POM (Hàng loạt)", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_clean_pom(t_file)
        if not data_test:
            st.warning(f"🚫 File không tách được POM chuẩn: {t_file.name}")
            continue

        with st.expander(f"🔍 ĐANG PHÂN TÍCH: {t_file.name}", expanded=True):
            v_test = get_vector(data_test['img'])
            
            # --- FIX LỖI TYPEERROR TẠI ĐÂY ---
            best_score, best_match = 0.0, None
            if samples and v_test:
                for s in samples:
                    # Lấy phần tử [0][0] của ma trận similarity
                    sim = float(cosine_similarity([v_test], [s['vector']])[0][0])
                    if sim > best_score:
                        best_score, best_match = sim, s
            
            if best_match:
                st.success(f"🤖 Đã khớp mã: **{best_match['file_name']}** (Dáng giống: {best_score:.1%})")
                c1, c2 = st.columns(2)
                with c1: st.image(data_test['img'], caption="BẢN VẼ TEST", use_container_width=True)
                with c2: st.image(best_match['img_url'], caption=f"MẪU GỐC: {best_match['file_name']}", use_container_width=True)
                
                # --- SO SÁNH POM ---
                st.write("### 📏 BẢNG ĐỐI CHIẾU THÔNG SỐ")
                res_comp = []
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                all_keys = sorted(set(list(p_t.keys()) + list(p_g.keys())))
                
                for k in all_keys:
                    v_t, v_g = p_t.get(k, "---"), p_g.get(k, "---")
                    status = "✅ KHỚP" if str(v_t).strip() == str(v_g).strip() else "❌ LỆCH"
                    if v_t != "---" or v_g != "---":
                        res_comp.append({"Hạng mục POM": k, "Test": v_t, "Gốc": v_g, "Kết quả": status})
                
                df_res = pd.DataFrame(res_comp)
                
                def style_result(val):
                    if val == "❌ LỆCH": return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'

                st.dataframe(df_res.style.map(style_result, subset=['Kết quả']), use_container_width=True)
                st.download_button(f"📥 Tải báo cáo {t_file.name}", df_res.to_csv(index=False).encode('utf-8-sig'), f"Report_{t_file.name}.csv", "text/csv")
            else:
                st.error(f"❌ Không tìm thấy mẫu tương đồng cho {t_file.name}")
