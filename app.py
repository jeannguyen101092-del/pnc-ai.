# ==========================================================
# AI FASHION AUDITOR V38.3 - POM DESCRIPTION & SIZE SELECTOR
# ==========================================================
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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V38.3", page_icon="📊")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- CÔNG CỤ TRÍCH XUẤT BẢNG THÔNG MINH ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip()
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_techpack_v383(pdf_file):
    """Trích xuất POM: Lấy Description làm Key, lưu tất cả các Size"""
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").upper()
                all_txt += txt + " "
                tables = page.extract_tables()
                for tb in tables:
                    if len(tb) < 2: continue
                    df = pd.DataFrame(tb)
                    
                    # 1. Tìm dòng Header (chứa chữ Description hoặc POM)
                    desc_col_idx = -1
                    size_cols = {} # Lưu {Tên Size: Index Cột}
                    
                    for row_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        if any("DESC" in s for s in row_up):
                            # Tìm vị trí cột Description
                            for i, val in enumerate(row):
                                val_up = str(val).upper()
                                if "DESC" in val_up: desc_col_idx = i
                                # Tìm các cột Size (Thường là tên ngắn: S, M, L, 28, 30...)
                                elif val and len(str(val)) <= 3 and val_up not in ["TOL", "NO", "CODE"]:
                                    size_cols[val_up] = i
                            
                            # Quét dữ liệu từ các dòng tiếp theo
                            for d_idx in range(row_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[desc_col_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or name == 'NAN': continue
                                
                                # Lưu thông số cho từng Size
                                if name not in full_specs: full_specs[name] = {}
                                for s_name, s_idx in size_cols.items():
                                    val = parse_val(d_row[s_idx])
                                    if val > 0: full_specs[name][s_name] = val
                            break
        
        # Nhận diện loại hàng
        cat = "QUẦN" if any(k in all_txt for k in ['INSEAM', 'RISE', 'PANT', 'WAIST']) else "ÁO"
        return {"specs": full_specs, "img": img_bytes, "cat": cat}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    files = st.file_uploader("Nạp Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP VÀO HỆ THỐNG"):
        p_bar = st.progress(0)
        p_text = st.empty()
        for i, f in enumerate(files):
            d = extract_techpack_v383(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "img_base64": base64.b64encode(d['img']).decode(), "category": d['cat']
                }, on_conflict="file_name").execute()
            p_bar.progress((i + 1) / len(files))
            p_text.text(f"Đang nạp: {int(((i+1)/len(files))*100)}%")
        st.success("✅ Đã nạp thành công!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V38.3")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = extract_techpack_v383(test_file)
    if target and target['img']:
        # 1. Chọn Size muốn so sánh
        available_sizes = set()
        for p in target['specs'].values():
            available_sizes.update(p.keys())
        
        sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", sorted(list(available_sizes)), index=0)
        
        # 2. Tìm mẫu trong kho
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                v_ref = np.array(i['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)) * 100
                matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in best:
                st.subheader(f"✨ Mẫu khớp: {m['name']} ({m['sim']:.1f}%) - Đang so sánh Size: {sel_size}")
                c1, c2 = st.columns(2)
                c1.image(target['img'], caption="File đang kiểm")
                c2.image(base64.b64decode(m['img']), caption="Mẫu trong kho")

                # Bảng đối soát
                diff = []
                for p_name, p_vals in target['specs'].items():
                    v1 = p_vals.get(sel_size, 0)
                    # Tìm thông số tương ứng trong mẫu kho (cùng hạng mục, cùng size)
                    v2 = 0
                    if p_name in m['spec']:
                        v2 = m['spec'][p_name].get(sel_size, 0)
                    
                    if v1 > 0 or v2 > 0:
                        diff.append({
                            "Hạng mục (Description)": p_name,
                            f"Thực tế ({sel_size})": v1,
                            f"Mẫu gốc ({sel_size})": v2,
                            "Chênh lệch": round(v1 - v2, 2)
                        })
                
                if diff:
                    df = pd.DataFrame(diff)
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                else:
                    st.warning(f"Không có dữ liệu cho Size {sel_size} trong mẫu này.")
