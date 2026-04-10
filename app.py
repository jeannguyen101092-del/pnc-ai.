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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.8", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- HÀM XỬ LÝ SỐ ĐO (HỖ TRỢ PHÂN SỐ 1/2, 3/4...) ---
def parse_measurement_val(t):
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

# --- TRÍCH XUẤT THÔNG SỐ ĐA DÒNG HÀNG (POM & DESCRIPTION) ---
def extract_pom_new_v438(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Lấy ảnh bìa làm thumbnail đối soát hình ảnh
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        # Nhận diện Brand
        all_text_full = ""
        for page in doc: all_text_full += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text_full: brand = "REITMANS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                # Chỉ quét các trang có từ khóa liên quan đến thông số (POM, Spec, Measurement)
                pg_txt = page.extract_text().upper() if page.extract_text() else ""
                if not any(k in pg_txt for k in ["POM", "SPEC", "MEASURE", "TOLERANCE", "SIZE"]):
                    continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_name_idx, val_idx = -1, -1
                    # Quét Header để tìm cột mô tả và cột số đo
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # 1. Tìm cột Tên thông số (Ưu tiên POM NAME của Reitmans, sau đó là Description)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "POINT OF MEASURE", "ITEM"]):
                                p_name_idx = i
                                break
                        
                        # 2. Tìm cột Giá trị (Ưu tiên NEW của Reitmans, sau đó là Final/Spec/Sample)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", "TOTAL"]):
                                val_idx = i
                                break
                        
                        # Nếu tìm thấy cặp cột phù hợp
                        if p_name_idx != -1 and val_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                # Lấy tên và làm sạch
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["REF:", "RELATED:", "DATE:", "PAGE"]): continue
                                
                                # Lấy giá trị
                                val_raw = d_row[val_idx]
                                val_num = parse_measurement_val(val_raw)
                                
                                if val_num > 0:
                                    full_specs[name] = val_num
                            break # Thoát vòng lặp row khi đã xử lý xong bảng này
                            
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# --- SIDEBAR: QUẢN LÝ THƯ VIỆN MẪU ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)
    except: st.warning("Chưa kết nối Supabase")

    files = st.file_uploader("Nạp Techpack mẫu (PDF)", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data: continue

            d = extract_pom_new_v438(f)
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

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V43.8 (Multi-Brand)")
t_file = st.file_uploader("Upload file cần ĐỐI SOÁT", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_new_v438(t_file)
    if target and target['specs']:
        st.success(f"✅ Brand nhận diện: **{target['brand']}** | Tìm thấy **{len(target['specs'])}** thông số.")
        
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
                with c1: st.image(target['img'], caption="Bản vẽ file đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc trong KHO")

                diff_list = []
                # So khớp thông số dựa trên tên đã làm sạch (Clean POS)
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)
                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                    
                    diff_list.append({
                        "Hạng mục (POM/Description)": p_name,
                        "Giá trị Kiểm tra": v_target,
                        "Giá trị Mẫu gốc": v_ref,
                        "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"
                    })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    # Highlight màu đỏ nếu lệch > 0.5 inch
                    def highlight_diff(val):
                        if isinstance(val, (int, float)) and abs(val) > 0.5:
                            return 'color: red; font-weight: bold'
                        return 'color: white'

                    st.table(df_r.style.applymap(highlight_diff, subset=['Lệch']))
                    
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Audit_{target['brand']}.xlsx")
            
            if st.button("🗑️ Xóa kết quả để quét file mới"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không tìm thấy bảng thông số. Hãy đảm bảo PDF có các cột như 'Description'/'POM Name' và 'New'/'Final'.")
