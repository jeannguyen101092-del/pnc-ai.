# ==========================================================
# AI FASHION AUDITOR V41.5 - ZERO ERROR & REITMANS STABLE
# ==========================================================
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Giữ nguyên URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V41.5", page_icon="📊")

# Quản lý xóa file tự động
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "audit_key" not in st.session_state: st.session_state.audit_key = 0

@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 để tiết kiệm RAM và ổn định nhất
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- XỬ LÝ SỐ ĐO PHÂN SỐ (14 1/2 -> 14.5) ---
def parse_val_fraction(t):
    try:
        t = str(t).replace(',', '.').strip().lower()
        if not t or t == 'nan': return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- TRÍCH XUẤT SIÊU MẠNH (ĐẶC TRỊ POM NAME) ---
def extract_techpack_v415(pdf_file):
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_page = (page.extract_text() or "").upper()
                all_txt += txt_page + " "
                tables = page.extract_tables()
                for tb in tables:
                    if len(tb) < 3: continue
                    df = pd.DataFrame(tb)
                    desc_idx = -1
                    size_cols = {}

                    # TÌM DÒNG HEADER
                    for idx, row in df.iterrows():
                        row_str = [str(c).upper().strip() for c in row if c]
                        if any(k in " ".join(row_str) for k in ["POM NAME", "DESCRIPTION", "POINT OF"]):
                            for i, val in enumerate(row):
                                val_s = str(val).upper().strip()
                                if any(x in val_s for x in ["POM NAME", "DESC"]): desc_idx = i
                                elif val and (val_s.isdigit() or (len(val_s) <= 3 and val_s not in ["TOL", "NO", "+", "-"])):
                                    size_cols[val_s] = i
                            
                            if desc_idx != -1:
                                for d_idx in range(idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    pos_name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                    if len(pos_name) < 3 or "SEE PAGE" in pos_name: continue
                                    if pos_name not in full_specs: full_specs[pos_name] = {}
                                    for s_name, s_idx in size_cols.items():
                                        v_num = parse_val_fraction(d_row[s_idx])
                                        if v_num > 0: full_specs[pos_name][s_name] = v_num
                            break
        return {"specs": full_specs, "img": img_bytes, "text": all_txt}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 HỆ THỐNG REITMANS")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Kho đang lưu trữ", f"{res_db.count} mẫu")
    except: pass

    files = st.file_uploader("Nạp Techpacks mẫu", accept_multiple_files=True, key=f"f_{st.session_state.uploader_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack_v415(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "category": "bottoms" if any(k in d for k in ["INSEAM", "WAIST", "PANT"]) else "tops"
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.session_state.uploader_key += 1
        st.success("✅ Nạp thành công!"); st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V41.5")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf", key=f"a_{st.session_state.audit_key}")

if test_file:
    target = extract_techpack_v415(test_file)
    if target and target['specs']:
        # 1. Danh sách Size Số
        all_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}), key=lambda x: int(x) if x.isdigit() else 0)
        sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", all_sizes)

        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                if i.get('vector'):
                    try:
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        # 🔥 FIX LỖI TYPEERROR DÒNG 140 TẠI ĐÂY: Thêm index [0][0]
                        sim_val = cosine_similarity(v_test, v_ref)[0][0]
                        sim = float(sim_val) * 100
                        matches.append({"data": i, "sim": sim})
                    except: continue
            
            top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            if top_matches:
                best = top_matches[0]
                st.subheader(f"✨ Khớp mẫu: {best['data']['file_name']} ({best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Mẫu đang kiểm")
                with c2: st.image(best['data']['image_url'], caption="Mẫu trong kho")

                diff_list = []
                for p_name, p_vals in target['specs'].items():
                    v1 = p_vals.get(sel_size, 0)
                    v2 = 0
                    # Khớp Description thông minh (Khớp 10 ký tự)
                    for k_ref, v_ref_map in best['data']['spec_json'].items():
                        if p_name[:10] in k_ref:
                            v2 = v_ref_map.get(sel_size, 0)
                            break
                    
                    if v1 > 0 or v2 > 0:
                        diff_list.append({
                            "Hạng mục (POM Name)": p_name,
                            f"Thực tế ({sel_size})": v1,
                            f"Mẫu gốc ({sel_size})": v2,
                            "Chênh lệch": round(v1 - v2, 2)
                        })
                
                if diff_list:
                    df_res = pd.DataFrame(diff_list)
                    st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                    
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df_res.to_excel(writer, index=False)
                    st.download_button(f"📥 Tải Excel {sel_size}", out.getvalue(), f"Audit_{sel_size}.xlsx")
            
            if st.button("🗑️ Reset để quét file mới"):
                st.session_state.audit_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không đọc được thông số. Vui lòng kiểm tra lại PDF Reitmans.")
