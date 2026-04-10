import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (BẮT BUỘC: Điền URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V46.1", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tol']): return 0
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

# --- TRÍCH XUẤT THÔNG SỐ (QUÉT CẠN POM, NÉ BOM) ---
def extract_pom_v461(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_pg = (page.extract_text() or "").upper()
                # NÉ TRANG BOM (Phụ liệu)
                if any(x in text_pg for x in ["POLY CORE", "SEWING THREAD", "BUTTON"]): continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        # Tìm cột Tên (Description/POM Name)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "ITEM", "POM #"]):
                                p_col = i
                                break
                        # Tìm cột Số đo (New/Size 12/Final/M)
                        for i, cell in enumerate(row_up):
                            if i != p_col and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " 12 ", " M "]):
                                v_col = i
                                break
                        
                        if p_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_col]).replace('\n',' ').strip().upper()
                                if len(name) < 2 or any(x in name for x in ["DATE", "PAGE", "REVISION"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: full_specs[name] = val
                            break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR: NẠP FILE (CÓ BẪY LỖI CHI TIẾT) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for idx, f in enumerate(files):
            d = extract_pom_v461(f)
            if d and d['specs']:
                img_url, vec = "", None
                try:
                    # 1. Xử lý Ảnh & Vector
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                    if model_ai:
                        img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                except: pass

                # 2. Lệnh nạp cưỡng bức và báo lỗi
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                        "image_url": img_url, "category": d['brand']
                    }).execute()
                    st.success(f"Đã nạp: {f.name}")
                except Exception as db_err:
                    st.error(f"❌ LỖI NẠP FILE {f.name}:")
                    st.code(str(db_err)) # HIỆN MÃ LỖI CHI TIẾT TẠI ĐÂY
            else:
                st.warning(f"Không quét được POM trong {f.name}")
            p_bar.progress((idx + 1) / len(files))
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46.1")
t_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v461(t_file)
    if target and target['specs']:
        st.success(f"✅ Quét được {len(target['specs'])} hạng mục thông số.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # AI tìm mẫu khớp nhất (ResNet18)
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim_val = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim_val})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Mẫu khớp: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc trong kho")

                diff_list = []
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)
                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                    diff_list.append({"Hạng mục": p_name, "Kiểm tra": v_target, "Mẫu gốc": v_ref, "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"})
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    # NÚT XUẤT EXCEL
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Audit_Report.xlsx")

if st.button("🗑️ Xóa kết quả"):
    st.session_state.au_key += 1
    st.rerun()
