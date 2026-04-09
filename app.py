import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V42.0", page_icon="👔")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- LOGIC NHẬN DIỆN THƯƠNG HIỆU ---
def detect_brand(text):
    text = text.upper()
    if "REITMANS" in text: return "REITMANS"
    if "EXPRESS" in text: return "EXPRESS"
    if "NIKE" in text: return "NIKE"
    return "OTHER"

# --- TRÍCH XUẤT V42 (LỌC GRADING & BRAND) ---
def extract_v42(pdf_file):
    full_specs, img_bytes, brand, all_txt = {}, None, "OTHER", ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        # Quét nhanh text toàn bộ để nhận diện Brand
        for page in doc:
            all_txt += page.get_text().upper() + " "
        brand = detect_brand(all_txt)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                
                # ĐIỀU KIỆN REITMANS: Chỉ quét nếu có GRADING NOT APPROVED
                if brand == "REITMANS" and "GRADING NOT APPROVED" not in txt_p:
                    continue
                
                if any(k in txt_p for k in ["POM NAME", "DESCRIPTION", "MEASURE"]):
                    tables = page.extract_tables()
                    for tb in tables:
                        df = pd.DataFrame(tb)
                        d_idx, s_cols = -1, {}
                        for r_idx, row in df.iterrows():
                            row_up = [str(c).upper().strip() for c in row if c]
                            if any(k in " ".join(row_up) for k in ["POM NAME", "DESC"]):
                                for i, v in enumerate(row):
                                    v_s = str(v).upper().strip()
                                    if "POM NAME" in v_s or "DESC" in v_s: d_idx = i
                                    elif v and (v_s.isdigit() or len(v_s) <= 3): s_cols[v_s] = i
                                
                                if d_idx != -1:
                                    for d_row_idx in range(r_idx + 1, len(df)):
                                        d_row = df.iloc[d_row_idx]
                                        name = str(d_row[d_idx]).replace('\n',' ').strip().upper()
                                        if len(name) < 3: continue
                                        if name not in full_specs: full_specs[name] = {}
                                        for s_n, s_i in s_cols.items():
                                            val = parse_val(d_row[s_i])
                                            if val > 0: full_specs[name][s_n] = val
                                break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://wikimedia.org", width=150)
    st.header("📂 HỆ THỐNG DỮ LIỆU")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_db.count if res_db.count else 0)

    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_v42(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                
                # Lưu thêm cột BRAND để ưu tiên tìm kiếm
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "category": d['brand'] # Dùng category để lưu Brand
                }, on_conflict="file_name").execute()
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V42.0")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_v42(t_file)
    if target and target['specs']:
        st.info(f"🏷️ Thương hiệu: **{target['brand']}**")
        
        # 1. ƯU TIÊN TÌM KIẾM THEO BRAND
        db_res = supabase.table("ai_data").select("*").eq("category", target['brand']).execute()
        
        # 2. Nếu cùng Brand không có, mới tìm sang các Brand khác
        if not db_res.data:
            st.warning(f"Không tìm thấy mẫu {target['brand']} nào. Đang tìm kiếm toàn bộ kho...")
            db_res = supabase.table("ai_data").select("*").execute()

        if db_res.data:
            all_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}))
            sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", all_sizes)

            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            with torch.no_grad():
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                v_ref = np.array(i['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)) * 100
                matches.append({"data": i, "sim": sim})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Khớp nhất: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Mẫu đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu trong kho")

                diff_list = []
                for p_name, p_vals in target['specs'].items():
                    v1 = p_vals.get(sel_size, 0)
                    v2 = m['data']['spec_json'].get(p_name, {}).get(sel_size, 0)
                    if v1 or v2:
                        diff_list.append({"Hạng mục (POM Name)": p_name, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": round(v1 - v2, 2)})
                
                if diff_list:
                    df = pd.DataFrame(diff_list)
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    # Xuất Excel
                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    st.download_button(f"📥 Tải Excel {sel_size}", out.getvalue(), f"Audit_{sel_size}.xlsx")
            
            if st.button("🗑️ Xóa để quét file mới"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ File không thỏa mãn điều kiện (Đối với Reitmans phải là trang GRADING NOT APPROVED).")
