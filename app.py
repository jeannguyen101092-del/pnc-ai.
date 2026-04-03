import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (HÃY ĐIỀN THÔNG TIN CỦA BẠN VÀO ĐÂY) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
GH_TOKEN = "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"
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

supabase: Client = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Fashion Pro V10", page_icon="👔")

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

# ================= GITHUB UPLOAD (ĐÃ FIX LINK) =================
def upload_to_github(img_bytes, filename):
    try:
        # Làm sạch tên file
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        # SỬA LỖI 1: Link API phải có ://github.com
        url = f"https://://github.com{GH_REPO}/contents/imgs/{clean_name}.jpg"
        
        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        content = base64.b64encode(img_bytes).decode('utf-8')
        
        # Kiểm tra file cũ để lấy SHA
        check = requests.get(url, headers=headers, timeout=10)
        data = {"message": f"Upload {clean_name}", "content": content, "branch": GH_BRANCH}
        if check.status_code == 200:
            data["sha"] = check.json()["sha"]
            
        # SỬA LỖI 2: Đẩy file lên GitHub
        res = requests.put(url, headers=headers, json=data, timeout=15)
        
        if res.status_code in [200, 201]:
            # SỬA LỖI 3: Trả về link RAW chuẩn để hiển thị được ảnh
            return f"https://githubusercontent.com{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.jpg"
        else:
            # Hiện lỗi trực tiếp để bạn biết vì sao không nạp được
            st.error(f"GitHub báo lỗi {res.status_code}: {res.text}")
            return None
    except Exception as e:
        st.error(f"Lỗi hệ thống GitHub: {e}")
        return None


# ================= TRÍCH XUẤT DỮ LIỆU =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE','SHOULDER']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r: continue
        txt = " | ".join([str(x) for x in r if x]).upper()
        if not any(k in txt for k in VALID_KEYS): continue
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        if not vals: continue
        specs[txt[:120]] = round(float(np.median(vals)), 2)
    return specs

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    inseam = 0
    length = 0
    # Tìm thông số để phân loại
    for k, v in specs.items():
        if 'INSEAM' in k: inseam = max(inseam, v)
        if 'LENGTH' in k: length = max(length, v)

    if 'CARGO' in txt: return "QUẦN CARGO"
    if 'ELASTIC' in txt: return "QUẦN LƯNG THUN"
    
    # FIX: Logic phân biệt quần dài và short chuẩn hơn
    if inseam >= 22 or length >= 30: return "QUẦN DÀI"
    if 0 < inseam <= 15 or 0 < length <= 22: return "QUẦN SHORT"
    
    if 'DRESS' in txt: return "ĐẦM"
    if 'SKIRT' in txt: return "VÁY"
    if 'SHIRT' in txt: return "ÁO SƠ MI"
    return "ÁO"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    specs.update(extract_specs(tb))
        
        doc = fitz.open(pdf_path)
        # Chụp ảnh trang 1 làm ảnh minh họa
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= GIAO DIỆN CHÍNH =================
st.title("👔 AI Fashion Pro V10")

with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    files = st.file_uploader("Upload PDF nạp kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_small = compress_image(d['img'])
                img_url = upload_to_github(img_small, f.name)
                if img_url:
                    # AI Vector
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(Image.open(io.BytesIO(img_small))).unsqueeze(0)).flatten().numpy().tolist()

                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
        st.success("Đã nạp xong!")
        st.rerun()

# ================= PHẦN SO SÁNH =================
test_file = st.file_uploader("Tải file Test đối chứng", type="pdf")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.info(f"Phân loại tìm được: **{target['cat']}**")
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()

            results = []
            for i in db.data:
                v_db = np.array(i['vector'])
                sim = float(cosine_similarity(v_test.reshape(1,-1), v_db.reshape(1,-1))[0][0]) * 100
                results.append({"name": i['file_name'], "sim": sim, "url": i['img_url']})
            
            results = sorted(results, key=lambda x: x['sim'], reverse=True)[:5]
            cols = st.columns(len(results))
            for i, res in enumerate(results):
                with cols[i]:
                    st.image(res['url'], caption=f"{res['name']} ({res['sim']:.1f}%)")
