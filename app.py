import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Hãy điền URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V46.0", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

# --- XỬ LÝ SỐ REITMANS (GIỮ NGUYÊN LOGIC CHUẨN) ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'tol']: return 0
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

# --- TRÍCH XUẤT THÔNG SỐ (NÂNG CẤP: NÉ BOM, VÉT SẠCH POM) ---
def extract_pom_deep_v46(pdf_file):
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

        # Từ khóa định danh bảng
        POM_KEYS = ["POM", "DESCRIPTION", "POINT", "MEASURE", "ITEM"]
        VALUE_KEYS = ["NEW", "SPEC", "FINAL", "SAMPLE", " M ", " L ", " 32 ", " 12 "]
        BOM_KEYS = ["POLY CORE", "SEWING THREAD", "BUTTON", "ZIPPER", "FABRIC", "LABEL"]

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_page = (page.extract_text() or "").upper()
                
                # NÉ TRANG BOM: Nếu thấy từ khóa phụ liệu thì bỏ qua trang này
                if any(k in text_page for k in BOM_KEYS): continue
                # ƯU TIÊN TRANG POM
                if not any(k in text_page for k in POM_KEYS): continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("").astype(str)
                    if df.empty or len(df.columns) < 2: continue
                    
                    pom_idx, val_idx = -1, -1
                    header_row = -1

                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in POM_KEYS): pom_idx = i
                            if any(k in cell for k in VALUE_KEYS): val_idx = i
                        
                        if pom_idx != -1 and val_idx != -1:
                            header_row = r_idx
                            break
                    
                    if header_row != -1:
                        for i in range(header_row + 1, len(df)):
                            row = df.iloc[i]
                            name = str(row[pom_idx]).replace("\n", " ").strip().upper()
                            # Lọc dòng rác và né dòng BOM rò rỉ
                            if len(name) < 3 or any(x in name for x in ["DATE", "PAGE", "NOTE", "THREAD"]): 
                                continue
                            
                            val = parse_reitmans_val(row[val_idx])
                            if val > 0: full_specs[name] = val
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR: NẠP FILE (CÓ THANH %) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Chưa kết nối Supabase")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_pom_deep_v46(f)
            if d and d['specs']:
                # Xử lý vector
                vec = None
                if model_ai:
                    img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # Upload ảnh & Nạp DB (Có bẫy lỗi để hiện chữ đỏ nếu k nạp đc)
                try:
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"x-upsert":"true"})
                    
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                    }).execute()
                except Exception as e:
                    st.error(f"Lỗi nạp file {f.name}: {e}")
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46.0")
t_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_deep_v46(t_file)
    if target and target['specs']:
        st.success(f"✅ Quét đúng trang POM: Tìm thấy **{len(target['specs'])}** hạng mục.")
        
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
                    sim_val = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim_val})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Mẫu khớp nhất: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc trong KHO")

                diff_list = []
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)
                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                    diff_list.append({
                        "Hạng mục": p_name,
                        "Kiểm tra (NEW)": v_target,
                        "Mẫu gốc (REF)": v_ref,
                        "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"
                    })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    # NÚT XUẤT EXCEL
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Report_{m['data']['file_name']}.xlsx")

if st.button("🗑️ Xóa kết quả"):
    st.session_state.au_key += 1
    st.rerun()
