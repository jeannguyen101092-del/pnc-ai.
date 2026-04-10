import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client

# --- CONFIG (Giữ nguyên thông tin của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V46", page_icon="🔍")

# --- MODEL AI (Dùng ResNet18 trích xuất đặc trưng ảnh) ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- UTILS ---
def parse_val(t):
    """Xử lý số Reitmans/Phân số: 1 1/2, 3/4..."""
    try:
        if t is None or str(t).lower() in ['nan', '', 'none', '-']: return 0
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def normalize_pom(t):
    """Làm sạch tên POM để soi đúng dòng dù Brand nào"""
    if not t: return ""
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s) # Chỉ giữ chữ và số để khớp tuyệt đối
    return s

def extract_pom_pro(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=io.BytesIO(pdf_content))
        if len(doc) > 0:
            # Tối ưu ảnh JPG cực nhẹ để AI soi nhạy hơn
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
            img_bytes = pix.tobytes("jpg", jpg_quality=50)
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        POM_KEYS = ["POM", "POINT", "MEASURE", "DESCRIPTION", "DIMENSION", "SPEC"]
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_page = (page.extract_text() or "").upper()
                if not any(k in text_page for k in POM_KEYS): continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    h_row, p_idx, v_idx = -1, -1, -1
                    for r_idx, row in df.head(8).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS): p_idx = i
                            if any(k in c for k in ["NEW", "SPEC", "SIZE", "SAMPLE"]): v_idx = i
                        if p_idx != -1 and v_idx != -1:
                            h_row = r_idx; break
                    
                    if h_row == -1: p_idx, v_idx = 0, 1

                    for i in range(h_row + 1, len(df)):
                        name = str(df.iloc[i][p_idx]).replace("\n", " ").strip().upper()
                        if len(name) < 3: continue
                        val = parse_val(df.iloc[i][val_idx])
                        if val > 0: full_specs[name] = float(val)
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

def get_img_vec(img_bytes):
    """Chuyển ảnh sang vector số để AI so sánh"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().numpy().astype(float).tolist()

# --- SIDEBAR: KHO DỮ LIỆU ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Số mẫu trong kho", res_c.count if res_c.count else 0)
    
    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True)
    if files and st.button("🚀 NẠP HỆ THỐNG"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_pom_pro(f)
            if d and d['specs']:
                vec = get_img_vec(d['img'])
                path = f"lib/{f.name.replace('.pdf', '.jpg')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/jpeg", "x-upsert":"true"})
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'], 
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }).execute()
            p.progress((i + 1) / len(files))
        st.success("✅ Đã nạp thành công!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46 PRO")
t_file = st.file_uploader("Upload file PDF cần kiểm tra", type="pdf")

if t_file:
    target = extract_pom_pro(t_file)
    if target and target['specs']:
        st.info(f"Đã quét được {len(target['specs'])} thông số. Đang đối soát...")
        
        db = supabase.table("ai_data").select("*").execute()
        if not db.data:
            st.error("Kho đang trống! Hãy nạp mẫu ở Sidebar trước.")
        else:
            v_test = np.array(get_img_vec(target['img']))
            matches = []
            for row in db.data:
                v_ref = np.array(row['vector'])
                sim = np.dot(v_test, v_ref) / (np.linalg.norm(v_test) * np.linalg.norm(v_ref))
                matches.append({"data": row, "sim": float(sim) * 100})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)
            if best and best[0]['sim'] > 60: # Ngưỡng tương đồng 60%
                m = best[0]
                st.success(f"🏆 Khớp mẫu: **{m['data']['file_name']}** ({m['sim']:.1f}%)")
                
                col1, col2 = st.columns(2)
                with col1: st.image(target['img'], caption="Bản vẽ Kiểm", use_container_width=True)
                with col2: st.image(m['data']['image_url'], caption="Mẫu gốc trong Kho", use_container_width=True)

                # --- SOI ĐÚNG DÒNG (ALIGNED) ---
                st.write("### 📊 Chi tiết đối soát thông số")
                ref_specs = m['data']['spec_json']
                ref_map = {normalize_pom(k): v for k, v in ref_specs.items()}
                
                diff_list = []
                for p_name, v_target in target['specs'].items():
                    p_key = normalize_pom(p_name)
                    v_ref = ref_map.get(p_key, 0)
                    diff = round(v_target - v_ref, 3)
                    
                    diff_list.append({
                        "Vị trí đo (POM)": p_name,
                        "Thực tế": v_target,
                        "Tiêu chuẩn": v_ref if v_ref > 0 else "N/A",
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                    })
                st.table(pd.DataFrame(diff_list))
            else:
                st.error("Không tìm thấy mẫu tương đồng. Hãy nạp lại kho dữ liệu!")

if st.button("RESET"): st.rerun()
