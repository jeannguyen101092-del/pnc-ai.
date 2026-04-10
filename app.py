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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V44.8", page_icon="📊")

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
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tbd']): return 0
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

# --- HÀM QUÉT CẠN THÔNG SỐ ---
def extract_pom_v448(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.0, 1.0)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables: continue
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_name_idx, val_idx = -1, -1
                    # Quét Header linh hoạt (15 dòng đầu mỗi bảng)
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        
                        # Tìm cột Tên: Thêm DESCRIPTION, POM #, ITEM
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "ITEM", "POM #", "POINT OF MEASURE"]):
                                p_name_idx = i
                                break
                        # Tìm cột Số đo: Thêm NEW, FINAL, SPEC, VALUE, M, L, S
                        for i, cell in enumerate(row_up):
                            if i != p_name_idx and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "VALUE", "SPEC", " M ", " L ", " S "]):
                                val_idx = i
                                break
                        
                        if p_name_idx != -1 and val_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                # Chấp nhận cả mã POM ngắn (như A1, B)
                                if len(name) < 2 or any(x in name for x in ["DATE", "PAGE", "REVISION"]): continue
                                
                                val_num = parse_val(d_row[val_idx])
                                if val_num > 0: full_specs[name] = val_num
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: pass

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_v448(f)
            if d and d['specs']:
                vec, url = None, ""
                try:
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    if model_ai:
                        img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                except: pass

                try:
                    supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": d['specs'], "image_url": url}).execute()
                    st.success(f"Đã nạp: {f.name} ({len(d['specs'])} dòng)")
                except Exception as db_err:
                    st.error(f"Lỗi DB: {db_err}")
            else:
                st.warning(f"Không tìm thấy bảng thông số trong: {f.name}")
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V44.8")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v448(t_file)
    if target and target['specs']:
        st.success(f"✅ Quét được {len(target['specs'])} hạng mục.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            matches = []
            if target['img'] and model_ai:
                img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)
                for i in db_res.data:
                    if i.get('vector'):
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        sim = float(cosine_similarity(v_test, v_ref)) * 100
                        matches.append({"data": i, "sim": sim})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1] if matches else [{"data": db_res.data[-1], "sim": 0}]
            for m in top:
                st.subheader(f"✨ Đối chiếu: {m['data']['file_name']}")
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
                    st.table(pd.DataFrame(diff_list).style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
