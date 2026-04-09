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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.0", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- HÀM LÀM SẠCH CHUỖI ---
def clean_text(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- XỬ LÝ SỐ (16 1/2 -> 16.5) ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt == 'nan': return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- TRÍCH XUẤT REITMANS (POM NAME & NEW ONLY) ---
def extract_reitmans_data(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                # Chỉ lấy trang GRADING NOT APPROVED
                if "GRADING NOT APPROVED" not in txt_p: continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    p_name_idx, new_idx = -1, -1
                    
                    # 1. Định vị cột POM Name và cột New
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if "POM NAME" in row_up and "NEW" in row_up:
                            for i, val in enumerate(row):
                                if str(val).upper().strip() == "POM NAME": p_name_idx = i
                                if str(val).upper().strip() == "NEW": new_idx = i
                            
                            # 2. Lấy dữ liệu 2 cột này
                            if p_name_idx != -1 and new_idx != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                    val_new = parse_val(d_row[new_idx])
                                    
                                    if len(name) > 2 and val_new > 0:
                                        full_specs[name] = val_new
                            break
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR: QUẢN LÝ KHO ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu hiện có", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp mẫu Reitmans mới", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            # 🔥 KIỂM TRA TRÙNG TÊN TRƯỚC KHI NẠP
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data:
                st.warning(f"Skipped: {f.name} (Đã có trong kho)")
                continue

            d = extract_reitmans_data(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": "REITMANS"
                }).execute()
                st.success(f"Nạp thành công: {f.name}")
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V43.0")
t_file = st.file_uploader("Upload file Reitmans cần kiểm tra", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_reitmans_data(t_file)
    if target and target['specs']:
        st.info(f"✅ Đã trích xuất **{len(target['specs'])}** hạng mục từ cột NEW.")
        
        # Chỉ so sánh với hàng Reitmans trong kho
        db_res = supabase.table("ai_data").select("*").eq("category", "REITMANS").execute()
        
        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                v_ref = np.array(i['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)) * 100
                matches.append({"data": i, "sim": sim})
            
            m = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for best in m:
                st.subheader(f"✨ Mẫu khớp nhất: {best['data']['file_name']} ({best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(best['data']['image_url'], caption="Mẫu gốc trong kho")

                # --- SO SÁNH CỘT NEW ---
                diff_list = []
                for p_name, v_new_target in target['specs'].items():
                    v_new_ref = 0
                    p_clean = clean_text(p_name)
                    # Khớp vị trí 100%
                    for k_ref, val_ref in best['data']['spec_json'].items():
                        if p_clean == clean_text(k_ref):
                            v_new_ref = val_ref
                            break
                    
                    if v_new_target > 0 or v_new_ref > 0:
                        diff_list.append({
                            "Vị trí (POM Name)": p_name,
                            "Thông số NEW (Kiểm)": v_new_target,
                            "Thông số NEW (Gốc)": v_new_ref,
                            "Chênh lệch": round(v_new_target - v_new_ref, 2)
                        })
                
                if diff_list:
                    df = pd.DataFrame(diff_list)
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                    
                    # Xuất Excel
                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    st.download_button(f"📥 Tải Excel đối soát NEW", out.getvalue(), "Audit_New_Only.xlsx")
            
            if st.button("🗑️ Xóa để quét tiếp"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không tìm thấy bảng đo. Hãy kiểm tra PDF có chữ 'GRADING NOT APPROVED' và cột 'NEW' hay không.")
