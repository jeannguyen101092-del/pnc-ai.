import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Kiểm tra kỹ URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V44.6", page_icon="📊")

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

# --- TRÍCH XUẤT THÔNG SỐ (VÉT SẠCH DỮ LIỆU) ---
def extract_pom_v446(pdf_file):
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
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "ITEM", "POM #"]):
                                p_name_idx = i
                                break
                        for i, cell in enumerate(row_up):
                            if i != p_name_idx and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "VALUE", "SPEC", " M ", " L "]):
                                val_idx = i
                                break
                        if p_name_idx != -1 and val_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 2: continue
                                val_num = parse_val(d_row[val_idx])
                                if val_num > 0: full_specs[name] = val_num
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR: NẠP FILE ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)
    except: st.error("Lỗi kết nối database")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_v446(f)
            if d and d['specs']:
                # 1. Xử lý ảnh & Vector (An toàn)
                vec, url = None, ""
                try:
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    if model_ai:
                        img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                except: pass

                # 2. Lưu vào bảng (Chỉ nạp những cột chắc chắn có)
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "vector": vec, 
                        "spec_json": d['specs'], 
                        "image_url": url
                    }).execute()
                    st.toast(f"Đã nạp thành công: {f.name}")
                except Exception as e:
                    st.error(f"Lỗi nạp file {f.name}: {e}")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V44.6")
t_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v446(t_file)
    if target and target['specs']:
        st.success(f"✅ Quét thành công {len(target['specs'])} hạng mục.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            matches = []
            # Nếu có ảnh, tính similarity
            if target['img'] and model_ai:
                img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)
                for i in db_res.data:
                    if i.get('vector'):
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                        matches.append({"data": i, "sim": sim})
            
            # Lấy mẫu khớp nhất hoặc mẫu cuối cùng nếu không tính được sim
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1] if matches else [{"data": db_res.data[-1], "sim": 0}]
            
            for m in top:
                st.subheader(f"✨ Mẫu đối chiếu: {m['data']['file_name']}")
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
