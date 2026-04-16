import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state:
    st.session_state['reset_key'] = 0

def display_logo(width=200):
    if os.path.exists("logo.png"):
        st.image("logo.png", width=width)
    else:
        st.markdown(f"<h1 style='color: #1E3A8A;'>PPJ GROUP</h1>", unsafe_allow_html=True)

# ================= 2. AI CORE ENGINE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0] if isinstance(match, list) else match
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. PDF EXTRACTION (LINH HOẠT) =================
def extract_pdf_flexible(file_content):
    all_specs, img_bytes = {}, None
    try:
        # --- 1. LUÔN LẤY HÌNH VẼ TRƯỚC (ƯU TIÊN) ---
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        try:
            paths = page.get_drawings()
            bboxes = [p["rect"] for p in paths if p["rect"].width < page.rect.width * 0.9]
            if bboxes:
                x0, y0 = min([b.x0 for b in bboxes]), min([b.y0 for b in bboxes])
                x1, y1 = max([b.x1 for b in bboxes]), max([b.y1 for b in bboxes])
                crop = fitz.Rect(max(0, x0-30), max(0, y0-30), min(page.rect.width, x1+30), min(page.rect.height, y1+30))
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop)
            else: pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        except: pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
        img_bytes = pix.tobytes("png")
        doc.close()

        # --- 2. THỬ LẤY THÔNG SỐ (NẾU CÓ) ---
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for tb in tables:
                        df = pd.DataFrame(tb).fillna("")
                        if df.empty or len(df.columns) < 2: continue
                        desc_col, size_cols = -1, {}
                        for r_idx in range(min(15, len(df))):
                            row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                            for i, v in enumerate(row):
                                if "DESCRIPTION" in v or "POM NAME" in v: desc_col = i; break
                            if desc_col != -1: break
                        if desc_col == -1: continue
                        for r_idx in range(min(15, len(df))):
                            row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                            for i, v in enumerate(row):
                                if i == desc_col or not v: continue
                                if any(x in v for x in ["TOL", "GRADE", "CODE", "+/-"]): continue
                                if len(v) <= 8 or v.isdigit() or v in ["XS","S","M","L","XL"]: size_cols[i] = v
                            if size_cols: break
                        if size_cols:
                            for s_col, s_name in size_cols.items():
                                temp_data = {}
                                for d_idx in range(len(df)):
                                    pom_text = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                    if len(pom_text) < 3 or any(x in pom_text.upper() for x in ["DESCRIPTION", "POM NAME"]): continue
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0: temp_data[pom_text] = val
                                if temp_data:
                                    if s_name not in all_specs: all_specs[s_name] = {}
                                    all_specs[s_name].update(temp_data)
        except: pass # Nếu lỗi trích xuất bảng thì bỏ qua, vẫn giữ img_bytes

        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. UI PPJ GROUP =================
with st.sidebar:
    display_logo(width=220)
    st.markdown("---")
    st.title("📂 MASTER REPOSITORY")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Total Synchronized SKUs", f"{res_count.count or 0} Models")
    
    st.divider()
    st.subheader("📥 Data Ingestion")
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE DATABASE"):
        with st.spinner("AI Processing..."):
            for f in new_files:
                c = f.read(); h = get_file_hash(c)
                check = supabase.table("ai_data").select("id").eq("id", h).execute()
                if check.data: continue
                data = extract_pdf_flexible(c)
                if data and data.get('img'): # Chỉ cần có hình là cho phép Up
                    supabase.storage.from_(BUCKET).upload(f"lib_{h}.png", data['img'], {"upsert":"true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(f"lib_{h}.png")
                    }).execute()
        st.rerun()

h_col1, h_col2 = st.columns([1, 4])
with h_col1: display_logo(width=120)
with h_col2: st.title("AI SMART AUDITOR PRO")

st.markdown("---")
file_audit = st.file_uploader("📤 Drag & Drop Tech-Pack for Auditing", type="pdf")

# ================= NÂNG CẤP BỘ LỌC NHẠY CAO (Dán vào đoạn xử lý Audit) =================
if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_flexible(a_bytes)
    
    if target and target.get("img"):
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # --- LOGIC BỘ LỌC ĐA TẦNG (HIGH SENSITIVITY) ---
            t_name = file_audit.name.upper()
            
            def get_sensitivity_score(row_name):
                row_name = str(row_name).upper()
                score = 1.0 # Điểm mặc định
                
                # 1. Cấp độ 1: Phân loại gốc (Quần vs Váy) - Quan trọng nhất
                if ("SKIRT" in t_name) == ("SKIRT" in row_name): score += 0.5
                if ("PANT" in t_name or "QUAN" in t_name) == ("PANT" in row_name or "QUAN" in row_name): score += 0.5
                
                # 2. Cấp độ 2: Chi tiết túi (Cargo, Pocket, Welt, Patch)
                keywords_pocket = ["CARGO", "POCKET", "WELT", "PATCH", "MO", "DAP"]
                for kw in keywords_pocket:
                    if (kw in t_name) and (kw in row_name): score += 0.3 # Cộng điểm nếu cùng loại túi
                
                # 3. Cấp độ 3: Chi tiết lưng (Elastic/Thun vs Non-elastic)
                keywords_waist = ["ELASTIC", "THUN", "BELT", "WAISTBAND", "LUNG"]
                for kw in keywords_waist:
                    if (kw in t_name) and (kw in row_name): score += 0.3
                
                # 4. Cấp độ 4: Kiểu dáng (Short vs Long/Trouser)
                if ("SHORT" in t_name) == ("SHORT" in row_name): score += 0.4
                
                return score

            # Áp dụng bộ lọc nhạy
            df_db['sensitivity_weight'] = df_db['file_name'].apply(get_sensitivity_score)
            # -----------------------------------------------

            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            
            # ĐIỂM CUỐI CÙNG = (Độ tương đồng hình ảnh) x (Trọng số chi tiết túi/lưng)
            df_db['final_score'] = df_db['sim'] * df_db['sensitivity_weight']
            
            top_3 = df_db.sort_values('final_score', ascending=False).head(3)
            
            # (Phần hiển thị UI tiếp theo giữ nguyên...)

            
            st.subheader("🎯 AI Image Matches")
            cols = st.columns(4)
            with cols[0]: st.image(target['img'], caption="FILE ĐANG AUDIT", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Độ giống: {row['sim']:.1%}", use_container_width=True)
                    if st.button(f"CHỌN MẪU {i+1}", key=f"btn_{idx}"): st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel', top_3.iloc[0].to_dict())
            st.success(f"**ĐỐI ỨNG VỚI MẪU:** {best['file_name']}")

            # --- KIỂM TRA ĐỂ HIỂN THỊ THÔNG SỐ ---
            if target.get("all_specs") and len(target["all_specs"]) > 0:
                st.subheader("📋 Measurement Comparison")
                all_exp = []
                for sz, specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        ref_s = best['spec_json'].get(sz, {})
                        rows = []
                        for pom, val in specs.items():
                            rv = ref_s.get(pom, 0)
                            rows.append({"Point": pom, "Target": val, "Ref": rv, "Diff": f"{val-rv:+.3f}"})
                            all_exp.append({"Size": sz, "Point": pom, "Target": val, "Ref": rv, "Diff": val-rv})
                        st.table(pd.DataFrame(rows))
                
                # Nút Excel chỉ hiện khi có thông số
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as wr:
                    pd.DataFrame(all_exp).to_excel(wr, index=False)
                st.download_button("📥 DOWNLOAD REPORT", buf.getvalue(), f"Audit_{best['file_name']}.xlsx")
            else:
                st.warning("⚠️ File mới không tìm thấy bảng thông số. Chỉ có thể so sánh hình ảnh tương đồng.")

    else:
        st.error("❌ Không thể xử lý hình ảnh từ file này.")
