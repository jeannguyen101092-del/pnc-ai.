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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.7", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- XỬ LÝ SỐ REITMANS ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT THÔNG SỐ NEW (UPGRADE MULTI BRAND) ---
def extract_pom_new_v437(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        # --- Lấy ảnh ---
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")

        # --- Detect brand ---
        all_text = ""
        for page in doc:
            all_text += (page.get_text() or "").upper() + " "

        if "REITMANS" in all_text:
            brand = "REITMANS"
        doc.close()

        # --- KEYWORDS ---
        POM_KEYS = ["POM", "POINT", "MEASURE", "DESCRIPTION", "DIMENSION"]
        VALUE_KEYS = ["NEW", "SPEC", "MEAS", "MEASURE", "GARMENT", "SIZE", "SAMPLE"]

        # --- Đọc bảng ---
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:

                text_page = (page.extract_text() or "").upper()

                # 🔥 Ưu tiên trang có POM
                if not any(k in text_page for k in POM_KEYS):
                    continue

                tables = page.extract_tables()

                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2:
                        continue

                    # --- Clean dataframe ---
                    df = df.fillna("").astype(str)

                    # --- Tìm header ---
                    header_row = -1
                    pom_idx, val_idx = -1, -1

                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]

                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS):
                                pom_idx = i
                            if any(k in c for k in VALUE_KEYS):
                                val_idx = i

                        if pom_idx != -1 and val_idx != -1:
                            header_row = r_idx
                            break

                    # --- Nếu không tìm thấy header rõ → thử fallback ---
                    if header_row == -1:
                        pom_idx = 0
                        val_idx = 1

                    # --- Đọc dữ liệu ---
                    for i in range(header_row + 1, len(df)):
                        row = df.iloc[i]

                        try:
                            name = str(row[pom_idx]).replace("\n", " ").strip().upper()
                            val_raw = row[val_idx]

                            if len(name) < 3:
                                continue

                            if any(x in name for x in ["REF", "NOTE", "TOL", "GRADE"]):
                                continue

                            val = parse_reitmans_val(val_raw)

                            if val > 0:
                                full_specs[name] = val

                        except:
                            continue

        return {"specs": full_specs, "img": img_bytes, "brand": brand}

    except Exception as e:
        print("ERROR EXTRACT:", e)
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data: continue

            d = extract_pom_new_v437(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }).execute()
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V43.7")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_new_v437(t_file)
    if target and target['specs']:
        st.success(f"✅ Tìm thấy **{len(target['specs'])}** hạng mục thông số.")
        
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    # 🔥 FIX CHÍNH TẠI ĐÂY: Lấy giá trị đầu tiên của ma trận similarity
                    sim_matrix = cosine_similarity(v_test, v_ref)
                    sim_val = float(sim_matrix[0][0]) * 100
                    matches.append({"data": i, "sim": sim_val})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Mẫu khớp: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc trong KHO")

                diff_list = []
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)
                    # Tìm thông số khớp 100% trong kho
                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                    if v_target > 0 or v_ref > 0:
                        diff_list.append({
                            "Hạng mục": p_name,
                            "NEW (Kiểm tra)": v_target,
                            "NEW (Mẫu gốc)": v_ref,
                            "Lệch": round(v_target - v_ref, 2)
                        })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    # Nút tải Excel báo cáo
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), "Audit_Report.xlsx")
            
            if st.button("🗑️ Xóa file này để quét file khác"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ AI không tìm thấy bảng thông số. Kiểm tra lại PDF có cột 'POM NAME' và 'NEW' không.")
