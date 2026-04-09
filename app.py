# ==========================================================
# AI FASHION AUDITOR V39.0 - CATEGORY FILTER & EXCEL EXPORT
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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V39.0", page_icon="📊")

# Từ điển phân loại của bạn
CATEGORY_MAP = {
 "tops": ["t-shirt", "shirt", "blouse", "hoodie", "tee", "top"],
 "bottoms": ["pants", "jeans", "shorts", "trouser", "denim", "pant"],
 "dress": ["dress", "gown"],
 "outerwear": ["jacket", "coat", "blazer"],
 "onepiece": ["jumpsuit", "romper"]
}

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- CÔNG CỤ ---
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

def identify_category(text_content):
    text_content = text_content.lower()
    for cat, keywords in CATEGORY_MAP.items():
        if any(k in text_content for k in keywords):
            return cat
    return "other"

def extract_techpack(pdf_file):
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
                    desc_col_idx = -1
                    size_cols = {}
                    for row_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        if any("DESC" in s for s in row_up):
                            for i, val in enumerate(row):
                                val_up = str(val).upper()
                                if "DESC" in val_up: desc_col_idx = i
                                elif val and len(str(val)) <= 3 and val_up not in ["TOL", "NO", "CODE"]:
                                    size_cols[val_up] = i
                            for d_idx in range(row_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[desc_col_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or name == 'NAN': continue
                                if name not in full_specs: full_specs[name] = {}
                                for s_name, s_idx in size_cols.items():
                                    val = parse_val(d_row[s_idx])
                                    if val > 0: full_specs[name][s_name] = val
                            break
        
        # Nhận diện Category dựa trên text và tên file
        category = identify_category(all_txt + " " + pdf_file.name)
        return {"specs": full_specs, "img": img_bytes, "cat": category}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", res_db.count if res_db.count else 0)
    except: pass

    files = st.file_uploader("Nạp Techpacks mới", accept_multiple_files=True)
    if files and st.button("🚀 NẠP VÀO HỆ THỐNG"):
        for f in files:
            d = extract_techpack(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "img_base64": base64.b64encode(d['img']).decode(), "category": d['cat']
                }, on_conflict="file_name").execute()
        st.success("✅ Đã nạp xong!"); st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V39.0")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = extract_techpack(test_file)
    if target and target['specs']:
        st.subheader(f"📂 Loại hàng nhận diện: {target['cat'].upper()}")
        
        # Lọc danh sách mẫu trong kho CÙNG LOẠI
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not res.data:
            st.error(f"❌ Không tìm thấy mẫu nào thuộc nhóm '{target['cat'].upper()}' trong kho để đối soát.")
        else:
            # Chọn Size
            available_sizes = set()
            for p in target['specs'].values(): available_sizes.update(p.keys())
            sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", sorted(list(available_sizes)), index=0)

            # Tính tương đồng
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)) * 100
                    matches.append({"data": i, "sim": sim})
            
            # Hiển thị Top 3 mẫu tương đồng nhất
            top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            st.write(f"### 📈 Top {len(top_matches)} mẫu tương đồng nhất:")
            
            for idx, m in enumerate(top_matches):
                with st.expander(f"Lựa chọn {idx+1}: {m['data']['file_name']} (Độ giống: {m['sim']:.1f}%)", expanded=(idx==0)):
                    c1, c2 = st.columns(2)
                    with c1: st.image(target['img'], caption="File đang kiểm")
                    with c2: st.image(base64.b64decode(m['data']['img_base64']), caption="Mẫu trong kho")

                    # Bảng đối soát
                    diff = []
                    for p_name, p_vals in target['specs'].items():
                        v1 = p_vals.get(sel_size, 0)
                        v2 = m['data']['spec_json'].get(p_name, {}).get(sel_size, 0)
                        if v1 > 0 or v2 > 0:
                            diff.append({
                                "Hạng mục (Vị trí)": p_name,
                                f"Thực tế ({sel_size})": v1,
                                f"Mẫu gốc ({sel_size})": v2,
                                "Chênh lệch": round(v1 - v2, 2)
                            })
                    
                    if diff:
                        df_res = pd.DataFrame(diff)
                        st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                        
                        # --- XUẤT EXCEL ---
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_res.to_excel(writer, index=False, sheet_name='Comparison')
                        
                        st.download_button(
                            label=f"📥 Tải báo cáo Excel (Lựa chọn {idx+1})",
                            data=output.getvalue(),
                            file_name=f"Audit_{m['data']['file_name']}_{sel_size}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    else:
        st.error("Không trích xuất được dữ liệu từ file PDF.")
