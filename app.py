import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (Vui lòng điền thông tin của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V7", page_icon="👖")

# ================= UTILS & PARSING =================
def parse_reitmans_val(t):
    try:
        if t is None: return 0
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

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model_ai = load_ai()

def get_vector(img_bytes):
    if not img_bytes: return [0.0]*512
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0)).view(-1).numpy()
            v = v / (np.linalg.norm(v) + 1e-7)
        return v.tolist()
    except: return [0.0]*512

def extract_data(pdf_file):
    img_bytes = None
    pdf_content = pdf_file.read()
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    tables = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tbs = page.extract_tables()
            for tb in tbs:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True).fillna("")
                tables.append(df)
    return {"img": img_bytes, "tables": tables}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN TRỊ KHO")
    res = supabase.table("ai_data").select("*", count="exact").execute()
    st.metric("Tổng mẫu", res.count if res.count else 0)
    
    files = st.file_uploader("NẠP KHO (PDF)", accept_multiple_files=True)
    if files and st.button("🚀 NẠP"):
        for f in files:
            d = extract_data(f)
            vec = get_vector(d['img'])
            # Lưu bảng đầu tiên của file làm mẫu chuẩn
            specs = d['tables'][0].to_dict('records') if d['tables'] else []
            img_url = ""
            if d['img']:
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                if not isinstance(img_url, str): img_url = getattr(img_url, "public_url", "")
            
            supabase.table("ai_data").insert({
                "file_name": f.name, "vector": vec, "spec_json": specs, "image_url": img_url
            }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# ================= MAIN =================
st.title("👖 AI Fashion Pro V7 PRO")

f_test = st.file_uploader("Upload PDF kiểm tra", type="pdf")

if f_test:
    target = extract_data(f_test)
    if target['tables']:
        # Giả định dùng bảng đầu tiên tìm thấy
        df_target = target['tables'][0]
        cols = df_target.columns.tolist()
        
        # Cho phép chọn size và cột vị trí
        c1, c2 = st.columns(2)
        with c1: pom_col = st.selectbox("Cột Vị trí (POM):", cols, index=0)
        with c2: size_col = st.selectbox("Chọn Size kiểm:", [c for c in cols if c != pom_col])

        # Tìm mẫu AI
        v_test = get_vector(target['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        matches = []
        for row in db.data:
            sim = cosine_similarity([v_test], [row['vector']])[0][0] if row['vector'] else 0
            matches.append({"row": row, "sim": sim})
        
        if matches:
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            ref_data = best['row']
            df_ref = pd.DataFrame(ref_data['spec_json'])

            st.success(f"🏆 TOP MẪU TƯƠNG ĐỒNG: {ref_data['file_name']} (Khớp {best['sim']*100:.1f}%)")

            # --- HIỂN THỊ HÌNH ẢNH 2 BÊN ---
            im_c1, im_c2 = st.columns(2)
            with im_c1: st.image(target['img'], caption="ẢNH KIỂM TRA", use_container_width=True)
            with im_c2: st.image(ref_data['image_url'], caption="ẢNH MẪU GỐC", use_container_width=True)

            # --- LOGIC SO SÁNH ĐÚNG DÒNG (SIDE BY SIDE) ---
            st.write("### 📊 CHI TIẾT THÔNG SỐ")
            
            # Chuẩn bị dữ liệu file Mẫu (Map theo tên POM)
            # Giả định cột đầu tiên của file mẫu là tên POM
            ref_map = {}
            ref_pom_col = df_ref.columns[0]
            for _, r_r in df_ref.iterrows():
                ref_map[str(r_r[ref_pom_col]).strip().upper()] = parse_reitmans_val(r_r.get(size_col, r_r.iloc[1]))

            compare_rows = []
            for _, r_t in df_target.iterrows():
                pom_name = str(r_t[pom_col]).strip()
                val_new = parse_reitmans_val(r_t[size_col])
                val_ref = ref_map.get(pom_name.upper(), 0)
                diff = round(val_new - val_ref, 3)

                compare_rows.append({
                    "VỊ TRÍ ĐO (POM)": pom_name,
                    "FILE KIỂM (NEW)": val_new,
                    "FILE MẪU (REF)": val_ref,
                    "CHÊNH LỆCH": diff,
                    "KẾT QUẢ": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })

            res_df = pd.DataFrame(compare_rows)
            
            # Tô màu để dễ nhìn
            def highlight_diff(s):
                return ['color: red' if s['CHÊNH LỆCH'] != 0 else '' for _ in s]

            st.table(res_df.style.apply(highlight_diff, axis=1))

            # Nút tải báo cáo
            csv = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Tải báo cáo (.csv)", csv, f"Audit_{ref_data['file_name']}.csv")

st.divider()
if st.button("LÀM MỚI HỆ THỐNG"): st.rerun()
