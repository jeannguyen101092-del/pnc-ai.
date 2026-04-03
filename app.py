import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (KIỂM TRA LẠI CÁC THÔNG SỐ NÀY) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
GH_TOKEN = "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"
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

supabase: Client = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Fashion Pro V9", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM NÉN ẢNH =================
def compress_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue()

# ================= GITHUB UPLOAD (ĐÃ SỬA API) =================
def upload_to_github(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        url = f"https://github.com{GH_REPO}/contents/imgs/{clean_name}.jpg"
        
        # DÒNG NÀY PHẢI THẲNG HÀNG VỚI DÒNG url Ở TRÊN
        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        content = base64.b64encode(img_bytes).decode('utf-8')
        
        check = requests.get(url, headers=headers, timeout=10)
        data = {"message": f"Upload {clean_name}", "content": content, "branch": GH_BRANCH}
        if check.status_code == 200:
            data["sha"] = check.json()["sha"]
            
        res = requests.put(url, headers=headers, json=data, timeout=15)
        if res.status_code in [200, 201]:
            # Trả về link Raw để Streamlit có thể hiển thị ảnh trực tiếp
            return f"https://githubusercontent.com{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.jpg"
        return None
    except Exception as e:
        st.error(f"Lỗi kết nối GitHub: {e}")
        return None

# ================= TRÍCH XUẤT DỮ LIỆU =================
def parse_val(t):
    try:
        # Lấy số, phân số hoặc số thập phân
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE','SHOULDER']
BLOCK_KEYS = ['SIZE','SEASON','DATE','#','FABRIC','COLOR']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r: continue
        txt = " | ".join([str(x) for x in r if x]).upper()
        if any(x in txt for x in BLOCK_KEYS): continue
        if not any(k in txt for k in VALID_KEYS): continue
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        if not vals: continue
        # Lấy giá trị lớn nhất trong hàng đó làm chuẩn (thường là size lớn nhất)
        specs[txt[:120]] = round(float(max(vals)), 2)
    return specs

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    
    # Lấy thông số Inseam và Length để phân biệt dài/ngắn
    inseam = 0
    total_length = 0
    for k, v in specs.items():
        if 'INSEAM' in k: inseam = max(inseam, v)
        if 'LENGTH' in k: total_length = max(total_length, v)

    if 'CARGO' in txt: return "QUẦN CARGO"
    if 'ELASTIC' in txt: return "QUẦN LƯNG THUN"
    
    # Logic chuẩn cho quần dài vs short
    if inseam >= 25 or total_length >= 35: return "QUẦN DÀI"
    if 0 < inseam <= 15 or 0 < total_length <= 22: return "QUẦN SHORT"
    
    if 'DRESS' in txt: return "ĐẦM"
    if 'SKIRT' in txt: return "VÁY"
    if 'SHIRT' in txt: return "ÁO SƠ MI"
    if any(k in txt for k in ['TEE', 'TOP', 'HOODIE', 'JACKET']): return "ÁO"
    return "KHÁC"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    specs.update(extract_specs(tb))
        
        if len(specs) < 2: return None
        
        doc = fitz.open(pdf_path)
        # Tăng Matrix lên 2.0 để ảnh rõ nét hơn
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0)).tobytes("png")
        doc.close()
        
        return {
            "spec": specs, 
            "img": img_bytes, 
            "cat": classify_logic(specs, text, os.path.basename(pdf_path))
        }
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
        return None

# ================= GIAO DIỆN & XỬ LÝ =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        count_res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{count_res.count if count_res.count else 0} mẫu")
    except: st.metric("Tổng mẫu trong kho", "Lỗi kết nối")

    st.divider()
    if "up_key" not in st.session_state: st.session_state.up_key = 0
    files = st.file_uploader("Upload PDF nạp kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")

    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for idx, f in enumerate(files):
            try:
                p_bar.progress((idx + 1) / len(files))
                name = re.sub(r'\s*\(\d+\)', '', f.name)
                
                with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
                d = get_data("tmp.pdf")
                
                if d:
                    img_small = compress_image(d['img'])
                    img_url = upload_to_github(img_small, name)
                    
                    if img_url:
                        # Xử lý Vector AI
                        tf = transforms.Compose([
                            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                        ])
                        with torch.no_grad():
                            img_pil = Image.open(io.BytesIO(img_small))
                            vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()

                        supabase.table("ai_data").upsert({
                            "file_name": name, "vector": vec, "spec_json": d['spec'],
                            "img_url": img_url, "category": d['cat']
                        }, on_conflict="file_name").execute()
                
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            except Exception as e:
                st.error(f"Lỗi file {f.name}: {e}")

        st.session_state.up_key += 1
        st.success("✅ Đã nạp kho xong!")
        st.rerun()

# ================= PHẦN SO SÁNH (GIAO DIỆN CHÍNH) =================
st.title("👔 AI Fashion Pro V9")
test_file = st.file_uploader("Tải file đối chứng (Test)", type="pdf")

if test_file:
    with open("test.pdf","wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")

    if target:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(target['img'], caption=f"Ảnh từ PDF Test - Loại: {target['cat']}")
        
        # Tìm trong database cùng Category
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            # So sánh AI
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()

            matches = []
            for item in db.data:
                if item.get('vector'):
                    sim = float(cosine_similarity(v_test.reshape(1,-1), np.array(item['vector']).reshape(1,-1))[0][0]) * 100
                    matches.append({"data": item, "sim": sim})
            
            # Sắp xếp độ giống nhau
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:5]
            
            st.subheader("Kết quả tìm kiếm mẫu tương đồng:")
            cols = st.columns(len(matches))
            for i, m in enumerate(matches):
                with cols[i]:
                    st.image(m['data']['img_url'], use_container_width=True)
                    st.write(f"**{m['data']['file_name']}**")
                    st.write(f"Độ giống: {m['sim']:.1f}%")
        else:
            st.warning("Không tìm thấy mẫu nào cùng loại trong kho.")
