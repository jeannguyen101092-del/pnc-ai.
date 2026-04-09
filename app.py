import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V41.0", page_icon="👔")

CATEGORY_MAP = {
 "tops": ["t-shirt", "shirt", "blouse", "hoodie", "tee", "top", "polo"],
 "bottoms": ["pants", "jeans", "shorts", "trouser", "denim", "pant", "legging"],
 "dress": ["dress", "gown", "skirt"],
 "outerwear": ["jacket", "coat", "blazer", "vest"],
 "onepiece": ["jumpsuit", "romper", "bodysuit"]
}

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- TOOLS ---
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

def extract_techpack(pdf_file):
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Lấy ảnh trang 1
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").upper()
                all_txt += txt + " "
                if any(k in txt for k in ["POM", "SPEC", "MEASURE"]):
                    tables = page.extract_tables()
                    for tb in tables:
                        if not tb or len(tb) < 2: continue
                        df = pd.DataFrame(tb)
                        desc_idx = -1
                        for row_idx, row in df.iterrows():
                            if any("DESC" in str(c).upper() for c in row if c):
                                for i, val in enumerate(row):
                                    if val and "DESC" in str(val).upper(): desc_idx = i
                                size_cols = {str(val).upper(): i for i, val in enumerate(row) 
                                             if val and len(str(val)) <= 4 and str(val).upper() not in ["TOL", "NO", "CODE", "DESC"]}
                                for d_idx in range(row_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                    if len(name) < 3 or name == 'NAN': continue
                                    if name not in full_specs: full_specs[name] = {}
                                    for s_name, s_idx in size_cols.items():
                                        val = parse_val(d_row[s_idx])
                                        if val > 0: full_specs[name][s_name] = val
                                break
        
        # Phân loại
        cat = "other"
        for c, keywords in CATEGORY_MAP.items():
            if any(k in (all_txt + pdf_file.name).lower() for k in keywords):
                cat = c; break

        return {"specs": full_specs, "img": img_bytes, "cat": cat}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO DỮ LIỆU")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu sẵn có", res_db.count if res_db.count else 0)

    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        prog = st.progress(0)
        status_text = st.empty()
        
        for i, f in enumerate(files):
            d = extract_techpack(f)
            
            # 1. BỘ LỌC FILE LỖI: Bỏ qua nếu rỗng ảnh hoặc rỗng thông số
            if not d or not d['img'] or not d['specs']:
                st.warning(f"⚠️ Bỏ qua file: {f.name} (Không tìm thấy bảng POM hoặc ảnh)")
                continue
            
            try:
                # 2. TRÍCH HÌNH ẢNH LƯU RIÊNG VÀO STORAGE
                img_path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(
                    path=img_path, 
                    file=d['img'], 
                    file_options={"content-type": "image/png", "x-upsert": "true"}
                )
                img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)

                # 3. TẠO VECTOR AI
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # 4. LƯU DATABASE (CHỈ LƯU LINK ẢNH, KHÔNG LƯU BASE64)
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": img_url, "category": d['cat']
                }, on_conflict="file_name").execute()
                
                status_text.text(f"✅ Đã nạp thành công: {f.name}")
            except Exception as e:
                st.error(f"❌ Lỗi khi nạp {f.name}: {e}")
            
            prog.progress((i + 1) / len(files))
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V41.0")
test_file = st.file_uploader("Upload file đối soát", type="pdf")

if test_file:
    target = extract_techpack(test_file)
    if target and target['specs']:
        st.info(f"📂 Nhận diện: **{target['cat'].upper()}**")
        
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db_res.data:
            st.error(f"❌ KHÔNG TÌM THẤY: Kho chưa có mẫu cùng loại '{target['cat'].upper()}'.")
        else:
            available_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}))
            sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", available_sizes)

            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)) * 100
                    matches.append({"data": i, "sim": sim})
            
            top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            for idx, m in enumerate(top_matches):
                with st.expander(f"Lựa chọn {idx+1}: {m['data']['file_name']} (Giống {m['sim']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    with c1: st.image(target['img'], caption="Mẫu đang kiểm")
                    # TẢI ẢNH TỪ LINK STORAGE (NHANH & NHẸ)
                    with c2: st.image(m['data']['image_url'], caption="Mẫu trong kho")

                    diff_list = []
                    for p_name, p_vals in target['specs'].items():
                        v1 = p_vals.get(sel_size, 0)
                        v2 = m['data']['spec_json'].get(p_name, {}).get(sel_size, 0)
                        if v1 or v2:
                            diff_list.append({"Hạng mục": p_name, f"Thực tế ({sel_size})": v1, f"Gốc ({sel_size})": v2, "Lệch": round(v1 - v2, 2)})
                    
                    if diff_list:
                        df_res = pd.DataFrame(diff_list)
                        st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                        
                        out = io.BytesIO()
                        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                            df_res.to_excel(writer, index=False)
                        st.download_button(f"📥 Tải Excel - Lựa chọn {idx+1}", out.getvalue(), f"Audit_{m['data']['file_name']}.xlsx")
    else:
        st.error("⚠️ File rác hoặc không trích xuất được thông số. Bỏ qua.")
