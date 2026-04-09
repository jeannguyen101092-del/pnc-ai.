import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V41.1", page_icon="👔")

# Từ điển phân loại nghiêm ngặt
CATEGORY_MAP = {
 "tops": ["t-shirt", "shirt", "blouse", "hoodie", "tee", "top", "polo", "tank"],
 "bottoms": ["pants", "jeans", "shorts", "trouser", "denim", "pant", "legging", "jean"],
 "dress": ["dress", "gown", "skirt", "vay", "dam"],
 "outerwear": ["jacket", "coat", "blazer", "vest"],
 "onepiece": ["jumpsuit", "romper", "bodysuit"]
}

@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 để tiết kiệm RAM (Tránh lỗi Over Resource)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- CÔNG CỤ XỬ LÝ ---
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

def identify_category(text_content, file_name):
    txt = (text_content + " " + file_name).lower()
    for cat, keywords in CATEGORY_MAP.items():
        if any(k in txt for k in keywords):
            return cat
    return "other"

def extract_techpack(pdf_file):
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Lấy ảnh trang 1 (Sketch)
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
                        # Tìm dòng tiêu đề chứa Description và các cột Size
                        for row_idx, row in df.iterrows():
                            row_up = [str(c).upper() for c in row if c]
                            if any("DESC" in s for s in row_up):
                                for i, val in enumerate(row):
                                    if val and "DESC" in str(val).upper(): desc_idx = i
                                
                                size_cols = {str(val).upper(): i for i, val in enumerate(row) 
                                             if val and len(str(val)) <= 5 and str(val).upper() not in ["TOL", "NO", "CODE", "DESC", "METHOD"]}
                                
                                for d_idx in range(row_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    # Lấy tên vị trí (Hạng mục)
                                    name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                    if len(name) < 3 or name == 'NAN': continue
                                    if name not in full_specs: full_specs[name] = {}
                                    for s_name, s_idx in size_cols.items():
                                        val = parse_val(d_row[s_idx])
                                        if val > 0: full_specs[name][s_name] = val
                                break
        return {"specs": full_specs, "img": img_bytes, "cat": identify_category(all_txt, pdf_file.name)}
    except: return None

# --- SIDEBAR: QUẢN LÝ KHO ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO DỮ LIỆU")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng mẫu sẵn có", res_db.count if res_db.count else 0)
    except: pass

    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        prog = st.progress(0)
        status_text = st.empty()
        for i, f in enumerate(files):
            d = extract_techpack(f)
            # BỘ LỌC FILE LỖI
            if not d or not d['img'] or not d['specs']:
                st.warning(f"⚠️ Bỏ qua: {f.name} (Không thấy POM/Ảnh)")
                continue
            
            try:
                # 1. Lưu ảnh vào Storage riêng
                img_path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=img_path, file=d['img'], file_options={"content-type": "image/png", "x-upsert": "true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)

                # 2. Tạo Vector
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # 3. Lưu Database
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": img_url, "category": d['cat']
                }, on_conflict="file_name").execute()
                status_text.text(f"✅ Đã nạp: {int(((i+1)/len(files))*100)}%")
            except Exception as e:
                st.error(f"❌ Lỗi nạp {f.name}: {e}")
            prog.progress((i + 1) / len(files))
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V41.1")
test_file = st.file_uploader("Upload file đối soát", type="pdf")

if test_file:
    target = extract_techpack(test_file)
    if target and target['specs']:
        st.info(f"📂 Nhận diện: **{target['cat'].upper()}**")
        
        # PHÂN LOẠI NGHIÊM NGẶT
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db_res.data:
            st.error(f"❌ KHÔNG TÌM THẤY: Kho chưa có mẫu cùng loại '{target['cat'].upper()}'.")
        else:
            # Lấy danh sách size thực tế từ bảng
            available_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}))
            if available_sizes:
                sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", available_sizes)

                # Tính tương đồng
                img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

                matches = []
                for i in db_res.data:
                    if i.get('vector'):
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        # FIX LỖI TYPEERROR TẠI ĐÂY: Lấy giá trị matrix [0][0]
                        sim_matrix = cosine_similarity(v_test, v_ref)
                        sim = float(sim_matrix[0][0]) * 100
                        matches.append({"data": i, "sim": sim})
                
                top_3 = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
                
                for idx, m in enumerate(top_3):
                    with st.expander(f"Lựa chọn {idx+1}: {m['data']['file_name']} (Giống {m['sim']:.1f}%)"):
                        c1, c2 = st.columns(2)
                        with c1: st.image(target['img'], caption="Mẫu đang kiểm")
                        with c2: st.image(m['data']['image_url'], caption="Mẫu trong kho")

                        diff_list = []
                        for p_name, p_vals in target['specs'].items():
                            v1 = p_vals.get(sel_size, 0)
                            v2 = m['data']['spec_json'].get(p_name, {}).get(sel_size, 0)
                            if v1 or v2:
                                diff_val = round(v1 - v2, 2)
                                diff_list.append({"Hạng mục": p_name, f"Thực tế ({sel_size})": v1, f"Gốc ({sel_size})": v2, "Lệch": diff_val})
                        
                        if diff_list:
                            df_res = pd.DataFrame(diff_list)
                            st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df_res.to_excel(writer, index=False)
                            st.download_button(f"📥 Tải Excel - {m['data']['file_name']}", output.getvalue(), f"Audit_{sel_size}.xlsx")
            else:
                st.warning("⚠️ Không tìm thấy cột thông số Size trong bảng POM.")
    else:
        st.error("⚠️ File rác hoặc PDF không có bảng POM hợp lệ.")
