import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (ĐIỀN LẠI THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
GH_TOKEN = "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"
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V10", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM UPLOAD GITHUB (ĐÃ FIX LỖI DÍNH CHỮ URL) =================
def upload_to_github(img_bytes, filename):
    try:
        # Làm sạch tên file và đảm bảo đường dẫn repo chuẩn
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        repo_path = GH_REPO.strip("/") # Xóa dấu / thừa nếu có
        
        # SỬA LỖI CHÍ MẠNG: Thêm dấu / sau ://github.com
        url = f"https://://github.com{repo_path}/contents/imgs/{clean_name}.jpg"
        
        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        content = base64.b64encode(img_bytes).decode('utf-8')
        
        # Kiểm tra file cũ để lấy SHA
        check = requests.get(url, headers=headers, timeout=10)
        data = {"message": f"Up {clean_name}", "content": content, "branch": GH_BRANCH}
        if check.status_code == 200:
            data["sha"] = check.json()["sha"]
            
        res = requests.put(url, headers=headers, json=data, timeout=15)
        
        if res.status_code in [200, 201]:
            # Link RAW chuẩn để Streamlit hiển thị được ảnh
            return f"https://githubusercontent.com{repo_path}/{GH_BRANCH}/imgs/{clean_name}.jpg"
        else:
            st.error(f"❌ GitHub báo lỗi {res.status_code}: {res.text}")
            return None
    except Exception as e:
        st.error(f"❌ Lỗi hệ thống GitHub: {e}")
        return None

# ================= TRÍCH XUẤT & PHÂN LOẠI (SỬA LỖI QUẦN DÀI/SHORT) =================
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

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    inseam = 0
    length = 0
    for k, v in specs.items():
        if 'INSEAM' in k: inseam = max(inseam, v)
        if 'LENGTH' in k: length = max(length, v)

    # LOGIC MỚI: Ưu tiên Quần Dài nếu Inseam hoặc Length đủ lớn
    if 'CARGO' in txt: return "QUẦN CARGO"
    if inseam >= 22 or length >= 30: return "QUẦN DÀI"
    if 0 < inseam <= 15 or 0 < length <= 22: return "QUẦN SHORT"
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
                    for r in tb:
                        if not r: continue
                        txt_r = " | ".join([str(x) for x in r if x]).upper()
                        if any(k in txt_r for k in ['INSEAM','WAIST','HIP','LENGTH','CHEST']):
                            vals = [parse_val(x) for x in r if x and parse_val(x) > 0]
                            if vals: specs[txt_r[:100]] = round(float(np.median(vals)), 2)
        
        doc = fitz.open(pdf_path)
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
                # Nén ảnh JPEG
                img_pil = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO()
                img_pil.save(buf, format="JPEG", quality=80)
                img_small = buf.getvalue()
                
                # Upload
                img_url = upload_to_github(img_small, f.name)
                
                if img_url:
                    # AI Vector
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()

                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
                    st.toast(f"✅ Xong: {f.name}")
        st.success("🏁 Hoàn tất nạp kho!")
        if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.rerun()

# ================= PHẦN TEST =================
test_file = st.file_uploader("Tải file Test", type="pdf")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện: {target['cat']}")
        st.image(target['img'], width=300)
        
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        if db.data:
            st.write(f"Tìm thấy {len(db.data)} mẫu tương đồng.")
