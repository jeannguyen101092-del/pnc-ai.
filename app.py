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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V41.3", page_icon="📊")

# Quản lý xóa file tự động
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "audit_key" not in st.session_state: st.session_state.audit_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- XỬ LÝ PHÂN SỐ (Ví dụ: 13 1/4 -> 13.25) ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip()
        # Tìm các định dạng: "13 1/4", "13.25", "13"
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_techpack_v413(pdf_file):
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").upper()
                all_txt += txt + " "
                tables = page.extract_tables()
                for tb in tables:
                    if len(tb) < 2: continue
                    df = pd.DataFrame(tb)
                    desc_idx = -1
                    size_cols = {}
                    
                    # QUÉT HEADER THÔNG MINH
                    for row_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        # Tìm cột mô tả (Description hoặc POM Name)
                        if any(k in " ".join(row_up) for k in ["DESC", "POM NAME", "POM CODE", "POINT OF"]):
                            for i, val in enumerate(row):
                                val_s = str(val).upper().strip()
                                if any(x in val_s for x in ["DESC", "POM NAME"]): desc_idx = i
                                # Nhận diện Size: Là Số (24, 30) hoặc Chữ ngắn (S, M, L)
                                elif val and (val_s.isdigit() or (len(val_s) <= 3 and val_s not in ["TOL", "NO", "CODE"])):
                                    size_cols[val_s] = i
                            
                            if desc_idx != -1:
                                for d_idx in range(row_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    # Lấy tên vị trí thực tế
                                    name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                    if len(name) < 3 or name == 'NAN': continue
                                    
                                    if name not in full_specs: full_specs[name] = {}
                                    for s_name, s_idx in size_cols.items():
                                        val_num = parse_val(d_row[s_idx])
                                        if val_num > 0: full_specs[name][s_name] = val_num
                                break
        return {"specs": full_specs, "img": img_bytes, "text": all_txt}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_db.count if res_db.count else 0)

    files = st.file_uploader("Nạp Techpacks mới", accept_multiple_files=True, key=f"files_{st.session_state.uploader_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack_v413(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                img_path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=img_path, file=d['img'], file_options={"content-type": "image/png", "x-upsert": "true"})
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(img_path),
                    "category": "bottoms" if "INSEAM" in d or "WAIST" in d else "tops"
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.session_state.uploader_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V41.3")
test_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"audit_{st.session_state.audit_key}")

if test_file:
    target = extract_techpack_v413(test_file)
    if target and target['specs']:
        # 1. Hiện danh sách Size (24, 25, 26...)
        all_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}))
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
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)) * 100
                    matches.append({"data": i, "sim": sim})
            
            m = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            
            st.subheader(f"✨ Mẫu khớp nhất: {m['data']['file_name']} ({m['sim']:.1f}%)")
            c1, c2 = st.columns(2)
            c1.image(target['img'], caption="Mẫu đang kiểm")
            c2.image(m['data']['image_url'], caption="Mẫu trong kho")

            # SO SÁNH THÔNG SỐ (Lấy đúng tên vị trí)
            diff_list = []
            for p_name, p_vals in target['specs'].items():
                v1 = p_vals.get(sel_size, 0)
                v2 = m['data']['spec_json'].get(p_name, {}).get(sel_size, 0)
                if v1 or v2:
                    diff_list.append({"Hạng mục (Vị trí)": p_name, f"Kiểm ({sel_size})": v1, f"Gốc ({sel_size})": v2, "Lệch": round(v1 - v2, 2)})
            
            if diff_list:
                df_res = pd.DataFrame(diff_list)
                st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                
                # Xuất Excel
                out = io.BytesIO()
                df_res.to_excel(out, index=False)
                st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Audit_{sel_size}.xlsx")
            
            if st.button("🗑️ Xóa để quét file mới"):
                st.session_state.audit_key += 1
                st.rerun()
    else:
        st.error("⚠️ AI không đọc được bảng POM Name. Vui lòng kiểm tra lại PDF.")
