import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Hãy kiểm tra kỹ URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45.2", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

# --- XỬ LÝ SỐ REITMANS (GIỮ NGUYÊN) ---
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

# --- TRÍCH XUẤT (VÉT SẠCH THÔNG SỐ - NÉ TRANG BOM) ---
def extract_pom_deep_v452(pdf_file):
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
                page_txt = page.extract_text().upper() if page.extract_text() else ""
                # Né trang phụ liệu (BOM) để không lấy nhầm chỉ may
                if "POLY CORE" in page_txt or "SEWING THREAD" in page_txt: continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_idx, v_idx = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # Ưu tiên Reitmans (POM NAME), sau đó là Description
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "ITEM"]):
                                p_idx = i
                                break
                        # Ưu tiên Reitmans (NEW), sau đó là Spec/Sample
                        for i, cell in enumerate(row_up):
                            if i != p_idx and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M "]):
                                v_idx = i
                                break
                            
                        if p_idx != -1 and v_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["DATE", "PAGE"]): continue
                                val = parse_reitmans_val(d_row[v_idx])
                                if val > 0: full_specs[name] = val
                            break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR (THANH % VÀ FIX NẠP) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Chưa kết nối Supabase")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_pom_deep_v452(f)
            if d and d['specs']:
                # 1. Tính vector (An toàn)
                vec = None
                if model_ai:
                    try:
                        img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                    except: pass

                # 2. Cố gắng nạp vào DB (Thử nạp tối giản nếu cột mới bị lỗi)
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "spec_json": d['specs'], 
                        "vector": vec, "category": d['brand']
                    }).execute()
                except:
                    # Nạp bản tối giản nếu DB thiếu cột category/vector
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "spec_json": d['specs']
                    }).execute()
                    
            p_bar.progress((i + 1) / len(files))
        st.success("Hoàn thành!")
        st.rerun()

# --- MAIN (SO SÁNH AI & NÚT EXCEL) ---
st.title("🔍 AI Fashion Auditor V45.2")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_deep_v452(t_file)
    if target and target['specs']:
        st.success(f"✅ Quét thành công **{len(target['specs'])}** hạng mục.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            
            diff_list = []
            for p_name, v_target in target['specs'].items():
                v_ref = m['spec_json'].get(p_name, 0)
                # Nếu không khớp tên chính xác, thử làm sạch tên để đối chiếu
                if v_ref == 0:
                    p_clean = clean_pos(p_name)
                    for k_ref, val_ref in m['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                
                diff_list.append({
                    "Hạng mục": p_name, "Kiểm tra": v_target, 
                    "Mẫu gốc": v_ref, "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"
                })
            
            if diff_list:
                df_r = pd.DataFrame(diff_list)
                st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                
                # NÚT XUẤT EXCEL
                out = io.BytesIO()
                df_r.to_excel(out, index=False)
                st.download_button("📥 Tải báo cáo Excel", out.getvalue(), "Report.xlsx")
