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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V42.8", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- HÀM LÀM SẠCH CHUỖI CỰC MẠNH ---
def clean_text_v428(t):
    # Chỉ giữ chữ cái và số, xóa sạch khoảng trắng và ký tự lạ để khớp 100%
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

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

def detect_brand(text):
    text = text.upper()
    if "REITMANS" in text: return "REITMANS"
    return "OTHER"

# --- TRÍCH XUẤT V42.8 (FIX LỖI MẤT DÒNG) ---
def extract_data(pdf_file):
    full_specs, img_bytes, brand, all_txt = {}, None, "OTHER", ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        for page in doc: all_txt += (page.get_text() or "").upper() + " "
        brand = detect_brand(all_txt)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                grading_check = re.search(r"GRADING\s*NOT\s*APPROVED", txt_p)
                if brand == "REITMANS" and not grading_check: continue
                
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
                                for data_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[data_idx]
                                    name = str(d_row[d_idx]).replace('\n',' ').strip().upper()
                                    # 🔥 SỬA LỖI: Cho phép tên ngắn (như HIP) và không chặn MASTER/REF
                                    if len(name) < 2 or name == 'NAN': continue
                                    
                                    if name not in full_specs: full_specs[name] = {}
                                    for s_n, s_i in s_cols.items():
                                        v_num = parse_val(d_row[s_i])
                                        if v_num > 0: full_specs[name][s_n] = v_num
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng số mẫu", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp mẫu Reitmans", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_data(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }, on_conflict="file_name").execute()
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V42.8")
t_file = st.file_uploader("Upload file kiểm tra", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_data(t_file)
    if target and target['specs']:
        st.success(f"🏷️ Brand: **{target['brand']}** | Tìm thấy **{len(target['specs'])}** hạng mục đo.")
        
        db_res = supabase.table("ai_data").select("*").eq("category", target['brand']).execute()
        if not db_res.data: db_res = supabase.table("ai_data").select("*").execute()

        if db_res.data:
            all_sz = sorted(list({s for p in target['specs'].values() for s in p.keys()}))
            sel_sz = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", all_sz)

            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim_m = cosine_similarity(v_test, v_ref)
                    matches.append({"data": i, "sim": float(sim_m) * 100})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Khớp mẫu: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc trong kho")

                diff_list = []
                for p_name, p_vals in target['specs'].items():
                    v1 = p_vals.get(sel_sz, 0)
                    v2 = 0
                    p_name_clean = clean_text_v428(p_name)
                    # 🔥 KHỚP DÒNG THÔNG MINH
                    for k_ref, v_ref_map in m['data']['spec_json'].items():
                        ref_clean = clean_text_v428(k_ref)
                        if p_name_clean in ref_clean or ref_clean in p_name_clean:
                            v2 = v_ref_map.get(sel_sz, 0)
                            break
                    if v1 > 0 or v2 > 0:
                        diff_list.append({"Hạng mục": p_name, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": round(v1 - v2, 2)})
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button(f"📥 Tải Excel {sel_sz}", out.getvalue(), f"Audit_{sel_sz}.xlsx")
            
            if st.button("🗑️ Xóa file vừa quét"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không tìm thấy bảng đo. Hãy kiểm tra PDF có chữ GRADING NOT APPROVED.")
