import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
# từ supabase import create_client (Đảm bảo bạn đã cài đặt: pip install supabase)

# ================= 1. CONFIGURATION =================
# Điền thông tin Supabase của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
# supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None

# ================= 2. AI CORE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            return (vec / np.linalg.norm(vec)).tolist()
    except: return None

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER (QUÉT TOÀN BỘ TRANG) =================
def extract_full_data(file_content):
    all_specs = {}
    all_imgs = []
    
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        for page in pdf.pages:
            # Lưu ảnh từng trang
            all_imgs.append(page.to_image(resolution=100).original)
            
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2: continue
                
                # Tìm dòng chứa Size (Ví dụ: XS, S, M hoặc số 2, 4, 6)
                header_idx = -1
                for i, row in enumerate(table):
                    row_str = " ".join([str(c) for c in row if c]).lower()
                    if any(x in row_str for x in ['pom', 'size', 'tol', 'description']):
                        header_idx = i
                        break
                
                if header_idx == -1: continue
                
                headers = table[header_idx]
                for row in table[header_idx+1:]:
                    if not row or not row[0]: continue
                    pom_name = str(row[0]).strip() + " " + str(row[1] if len(row)>1 else "")
                    
                    # Lọc bỏ bảng nguyên phụ liệu
                    if any(k in pom_name.lower() for k in ['bag', 'poly', 'thread', 'label', 'zipper']):
                        continue

                    for col_idx, size_name in enumerate(headers):
                        if not size_name or col_idx < 2: continue
                        size_name = str(size_name).strip().upper()
                        
                        # Chấp nhận size là chữ hoặc số đơn thuần
                        if len(size_name) > 5: continue 

                        val_raw = str(row[col_idx]).strip()
                        # Parse phân số (1/2, 1/4...)
                        try:
                            if "/" in val_raw:
                                parts = val_raw.split()
                                val = sum(float(f.split('/')[0])/float(f.split('/')[1]) if '/' in f else float(f) for f in parts)
                            else:
                                val = float(val_raw)
                            
                            if size_name not in all_specs: all_specs[size_name] = {}
                            all_specs[size_name][pom_name] = val
                        except: continue
                        
    return {"all_specs": all_specs, "imgs": all_imgs}

# ================= 4. MAIN INTERFACE =================
menu = ["🔍 Tìm kiếm tương đồng", "🔄 Version Control"]
mode = st.sidebar.selectbox("Chức năng", menu)

if mode == "🔍 Tìm kiếm tương đồng":
    st.subheader("🔍 Tìm kiếm mẫu tương đồng (AI Match)")
    up_file = st.file_uploader("Tải lên File PDF hoặc Ảnh", type=["pdf", "jpg", "png"])
    if up_file:
        st.info("Chức năng tìm kiếm yêu cầu kết nối Database Supabase để đối soát Vector.")

elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh dữ liệu giữa 2 File PDF")
    col1, col2 = st.columns(2)
    f1 = col1.file_uploader("File A (Gốc)", type="pdf", key="f1")
    f2 = col2.file_uploader("File B (Mới)", type="pdf", key="f2")
    
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh toàn bộ các trang", use_container_width=True):
            with st.spinner("Đang quét dữ liệu..."):
                st.session_state['ver_results'] = {
                    "d1": extract_full_data(f1.getvalue()),
                    "d2": extract_full_data(f2.getvalue()),
                    "n1": f1.name, "n2": f2.name
                }

    if st.session_state['ver_results']:
        res = st.session_state['ver_results']
        
        with st.expander("🖼️ Xem các trang đã quét"):
            t1, t2 = st.tabs(["File A", "File B"])
            t1.image(res['d1']['imgs'], caption=[f"A-Trang {i+1}" for i in range(len(res['d1']['imgs']))], width=200)
            t2.image(res['d2']['imgs'], caption=[f"B-Trang {i+1}" for i in range(len(res['d2']['imgs']))], width=200)

        all_sizes = sorted(list(set(res['d1']['all_specs'].keys()) | set(res['d2']['all_specs'].keys())))
        
        dfs, names = [], []
        for sz in all_sizes:
            with st.expander(f"📏 Size: {sz}", expanded=True):
                s1, s2 = res['d1']['all_specs'].get(sz, {}), res['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                
                rows = []
                for p in poms:
                    v1, v2 = s1.get(p, 0), s2.get(p, 0)
                    diff = v2 - v1
                    status = "✅ Khớp" if diff == 0 else "⚠️ Lệch"
                    rows.append({"POM": p, "File A": v1, "File B": v2, "Diff": f"{diff:+.3f}", "Status": status})
                
                df = pd.DataFrame(rows)
                st.table(df)
                dfs.append(df); names.append(sz)
        
        st.download_button("📥 Xuất Excel So Sánh", to_excel(dfs, names), "Comparison.xlsx")
