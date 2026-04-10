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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.9", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- HÀM XỬ LÝ SỐ REITMANS & CHUNG ---
def parse_val(t):
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

# --- TRÍCH XUẤT THÔNG SỐ ĐA DÒNG HÀNG ---
def extract_pom_new_v439(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        elif "EXPRESS" in all_text: brand = "EXPRESS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                pg_txt = page.extract_text().upper() if page.extract_text() else ""
                # Chỉ quét trang có từ khóa kỹ thuật
                if not any(k in pg_txt for k in ["POM", "SPEC", "MEASURE", "TOLERANCE", "SIZE", "DESCRIPTION"]):
                    continue

                               tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_name_idx, val_idx = -1, -1
                    # Quét Header để tìm cột
                    for r_idx, row in df.head(10).iterrows(): 
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # Tìm cột Tên (Linh hoạt hơn)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "ITEM", "POINT OF MEASURE"]):
                                p_name_idx = i
                                break
                        # Tìm cột Giá trị
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", "TOTAL", "VALUE"]):
                                val_idx = i
                                break
                        
                        if p_name_idx != -1 and val_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                
                                # Loại bỏ các dòng tiêu đề phụ hoặc dòng trống
                                if len(name) < 2 or any(x in name for x in ["DATE", "PAGE", "REVISION", "NOTE"]): 
                                    continue
                                
                                val_num = parse_val(d_row[val_idx])
                                # Lấy cả các thông số nhỏ (chấp nhận giá trị > 0)
                                if val_num > 0: 
                                    full_specs[name] = val_num
                            # KHÔNG dùng break ở đây để nó quét tiếp các cột/bảng khác nếu có


# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)
    except: pass

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data: continue

            d = extract_pom_new_v439(f)
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
st.title("🔍 AI Fashion Auditor V43.9")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_new_v439(t_file)
    if target and target['specs']:
        st.success(f"✅ Brand: {target['brand']} | Tìm thấy **{len(target['specs'])}** hạng mục.")
        
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
                st.subheader(f"✨ Mẫu khớp: {m['data']['file_name']} ({m['sim']:.1f}%)")
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
                        "Kiểm tra": v_target,
                        "Mẫu gốc": v_ref,
                        "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"
                    })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    
                    # Sửa lỗi AttributeError: Sử dụng .map() thay cho .applymap()
                    def style_diff(val):
                        try:
                            if isinstance(val, (int, float)) and abs(val) > 0.5:
                                return 'color: red; font-weight: bold'
                        except: pass
                        return 'color: white'

                    st.table(df_r.style.map(style_diff, subset=['Lệch']))
                    
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), "Audit_Report.xlsx")
            
            if st.button("🗑️ Xóa để quét file khác"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ AI không tìm thấy bảng thông số. Kiểm tra lại trang POM trong PDF.")
