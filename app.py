# ==========================================================
# AI FASHION PRO V8.3 - SIÊU NHẬN DIỆN (FIX ÁO/QUẦN)
# ==========================================================
import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.3", page_icon="👔")

# ================= AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ================= PHÂN LOẠI THÔNG MINH (TRỌNG TÂM) =================
def smart_classify(specs, all_text, file_name):
    """Quyết định loại hàng dựa trên cả POM và Văn bản"""
    txt = (str(all_text) + " " + str(file_name)).upper()
    
    # 1. Các từ khóa chắc chắn là QUẦN
    bottom_keywords = ['INSEAM', 'WAIST', 'HIP', 'RISE', 'LEG OPENING', 'THIGH', 'KNEE', 'TROUSER', 'PANT', 'SHORT', 'JEAN']
    # 2. Các từ khóa chắc chắn là ÁO
    top_keywords = ['CHEST', 'BUST', 'NECK', 'SLEEVE', 'SHOULDER', 'SHIRT', 'JACKET', 'HOODIE', 'TEE']

    # Kiểm tra trong bảng thông số (Specs)
    spec_keys = " ".join(specs.keys()).upper()
    
    # Ưu tiên Inseam/Rise để xác định là Quần
    if any(k in spec_keys for k in ['INSEAM', 'RISE', 'LEG OPENING']) or 'PANT' in txt:
        if 'SHORT' in txt: return "QUẦN SHORT"
        return "QUẦN"
    
    if any(k in spec_keys for k in top_keywords) or any(k in txt for k in ['SHIRT', 'JACKET']):
        return "ÁO"
        
    return "KHÁC"

# ================= TRÍCH XUẤT NÂNG CAO =================
def extract_specs_v8(table):
    specs = {}
    # Từ khóa kỹ thuật cần giữ lại
    VALID = ['WAIST','HIP','INSEAM','THIGH','KNEE','LEG','RISE','CHEST','SHOULDER','LENGTH','SLEEVE']
    for r in table:
        if not r or len(r) < 2: continue
        row_str = " ".join([str(x) for x in r if x]).upper()
        
        # Nếu dòng chứa từ khóa kỹ thuật
        if any(k in row_str for k in VALID):
            # Lấy tất cả các số trong dòng
            vals = []
            for cell in r:
                found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(cell))
                if found:
                    v = found[0]
                    try:
                        num = eval(v.replace(' ', '+')) if '/' in v else float(v)
                        if num > 0: vals.append(num)
                    except: continue
            
            if vals:
                # Key là cột đầu tiên (Description)
                key = str(r[0]).strip().upper()[:60]
                specs[key] = round(float(np.median(vals)), 2)
    return specs

def get_full_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                all_texts += (p.extract_text() or "") + " "
                tables = p.extract_tables()
                for table in tables:
                    specs.update(extract_specs_v8(table))
        
        doc = fitz.open(pdf_path)
        # Tìm trang có Sketch hoặc lấy trang 1
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        # Phân loại thông minh hơn
        category = smart_classify(specs, all_texts, os.path.basename(pdf_path))
        
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": category}
    except Exception as e:
        st.error(f"Lỗi xử lý PDF: {e}")
        return None

# ================= SIDEBAR: NẠP KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    files = st.file_uploader("Upload Techpacks (PDF)", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_full_data("tmp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                img_p = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                with torch.no_grad():
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "img_base64": d['img_b64'],
                    "category": d['cat']
                }, on_conflict="file_name").execute()
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Đã nạp xong!")
        st.rerun()

# ================= MAIN AREA =================
st.title("👔 AI Fashion Pro V8.3")
test_file = st.file_uploader("Upload file cần đối soát", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_full_data("test.pdf")
    
    if target:
        st.success(f"Nhận diện loại hàng: {target['cat']}")
        
        # Tìm kiếm trong Database theo Category
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if db_res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_target = Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')
            with torch.no_grad():
                v_test = ai_brain(tf(img_target).unsqueeze(0)).flatten().numpy()

            matches = []
            for item in db_res.data:
                if item.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(item['vector'])])) * 100
                    matches.append({"data": item, "sim": sim})
            
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:5]
            
            for m in matches:
                with st.expander(f"{m['data']['file_name']} | Khớp: {m['sim']:.1f}%"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img_bytes'], caption="Mẫu đang kiểm")
                    c2.image(base64.b64decode(m['data']['img_base64']), caption="Mẫu trong kho")
                    
                    # So sánh bảng POM
                    diff = []
                    # Lấy Description (Hạng mục) làm gốc để so sánh
                    for p in target['spec']:
                        v1 = target['spec'][p]
                        # Tìm giá trị tương ứng ở mẫu gốc (khớp chuỗi Description)
                        v2 = next((m['data']['spec_json'][k] for k in m['data']['spec_json'] if p[:10] in k), 0)
                        diff.append({
                            "Hạng mục (POM)": p,
                            "Mẫu mới": v1,
                            "Mẫu cũ": v2,
                            "Lệch": round(v1 - v2, 2),
                            "Kết quả": "✅ OK" if abs(v1-v2) < 0.5 else "❌ LỆCH"
                        })
                    
                    res_df = pd.DataFrame(diff)
                    st.table(res_df)
                    
                    # Nút xuất Excel
                    output = io.BytesIO()
                    res_df.to_excel(output, index=False)
                    st.download_button("📥 Tải báo cáo Excel", output.getvalue(), f"Audit_{m['data']['file_name']}.xlsx")
        else:
            st.warning(f"Chưa có dữ liệu cho loại hàng '{target['cat']}' trong kho.")
