import streamlit as st
import fitz, os, io, pickle, torch, pdfplumber, re
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= CẤU HÌNH HỆ THỐNG =================
st.set_page_config(layout="wide", page_title="AI MASTER PRO V12.8", page_icon="🚀")

# 1. KẾT NỐI GOOGLE DRIVE QUA SERVICE ACCOUNT
def get_gdrive_service():
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_info)
        return build('drive', 'v3', credentials=creds)
    except:
        st.error("❌ Chưa cấu hình Secret 'gcp_service_account' trên Streamlit Cloud!")
        return None

drive_service = get_gdrive_service()
FOLDER_ID = 'ID_THU_MUC_CUA_BAN' # <--- THAY ID THƯ MỤC PNC_PDF VÀO ĐÂY
DB_LOCAL_PATH = 'db_v12.pkl'
device = torch.device("cpu") # Streamlit Cloud dùng CPU là đủ nhanh

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet50(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.to(device).eval()

ai_brain = load_ai()

# ================= LOGIC XỬ LÝ PDF (GIỮ NGUYÊN TỪ V12.7) =================
KEY_GROUPS = {
    "Waist": ["Waist Width", "Waist At Edge", "Waist At Top"],
    "Hip/Seat": ["Hip Width", "Hip 3\"", "Hip 4\"", "Seat Width"],
    "Inseam": ["Inseam", "Inner Leg"],
    "Leg Opening": ["Leg Opening", "Bottom Opening"],
    "Chest/Bust": ["Chest Width", "Across Bust", "Bust Width"],
    "Length": ["Body Length", "HPS to hem", "Total Length"],
    "Shoulder": ["Shoulder Width", "Across Shoulder"],
    "Sleeve": ["Sleeve Length"]
}

def parse_val(t):
    try:
        t = str(t).replace(',', '.')
        if ' ' in t: p = t.split(); return float(p[0]) + eval(p[1])
        if '/' in t: return eval(t)
        return float(t)
    except: return None

def process_pdf(pdf_stream):
    specs = {}; category = "UNKNOWN"
    try:
        with pdfplumber.open(pdf_stream) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages]).lower()
            if any(x in txt for x in ["jacket", "shirt", "top", "hoodie", "tee"]): category = "TOP"
            elif any(x in txt for x in ["pant", "jean", "short", "bottom"]): category = "BOTTOM"
            
            all_rows = []
            for p in pdf.pages:
                if p.extract_table(): all_rows.extend(p.extract_table())
            
            # Logic tìm size (giữ nguyên bản cũ của bạn)
            for r in all_rows:
                r_c = [str(x).strip() for x in r if x]; r_s = " ".join(r_c)
                for std, keys in KEY_GROUPS.items():
                    if any(k.lower() in r_s.lower() for k in keys):
                        val = parse_val(r_c[-1]) # Lấy cột cuối cùng làm mẫu
                        if val and val > 2: specs[std] = val

        doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5,1.5))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return {"spec": specs, "img": img, "cat": category}
    except: return None

# ================= QUẢN LÝ DỮ LIỆU DRIVE =================
def sync_with_drive():
    db = {}
    if os.path.exists(DB_LOCAL_PATH):
        with open(DB_LOCAL_PATH, "rb") as f: db = pickle.load(f)

    if drive_service:
        query = f"'{FOLDER_ID}' in parents and trashed = false and name contains '.pdf'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        drive_files = results.get('files', [])
        
        # 1. Xóa những gì không còn trên Drive
        drive_names = [f['name'] for f in drive_files]
        db = {name: data for name, data in db.items() if name in drive_names}

        # 2. Thêm file mới
        new_files = [f for f in drive_files if f['name'] not in db]
        if new_files:
            st.info(f"🔄 Đang đồng bộ {len(new_files)} mẫu mới từ Drive...")
            tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            for f_info in new_files:
                # Tải file từ Drive vào bộ nhớ RAM
                request = drive_service.files().get_media(fileId=f_info['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()
                
                fh.seek(0)
                data = process_pdf(fh)
                if data:
                    with torch.no_grad():
                        vec = ai_brain(tf(data['img'].convert('RGB')).unsqueeze(0)).flatten().numpy()
                    db[f_info['name']] = {"vector": vec, "img": data['img'], "spec": data['spec'], "cat": data['cat']}
            
            with open(DB_LOCAL_PATH, "wb") as f: pickle.dump(db, f)
    return db

# ================= GIAO DIỆN WEB =================
st.sidebar.title("🛡️ AI MASTER PRO")
if st.sidebar.button("🔄 Đồng bộ dữ liệu Drive"):
    st.session_state.db = sync_with_drive()
    st.success("Đã đồng bộ xong!")

if 'db' not in st.session_state:
    st.session_state.db = sync_with_drive()

st.sidebar.metric("📦 TỔNG MẪU TRONG KHO", len(st.session_state.db))

up = st.file_uploader("📤 Thả PDF cần kiểm tra vào đây", type="pdf")
if up:
    target = process_pdf(up)
    if target:
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            t_vec = ai_brain(tf(target['img'].convert('RGB')).unsqueeze(0)).flatten().numpy()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("🔍 Mẫu Đang Quét")
            st.image(target['img'], use_container_width=True)
            st.write(target['spec'])
        
        with c2:
            st.subheader("✅ Kết Quả So Khớp")
            res = []
            for name, data in st.session_state.db.items():
                sim = float(cosine_similarity(t_vec.reshape(1,-1), data['vector'].reshape(1,-1))) * 100
                if sim > 65:
                    res.append({"name": name, "sim": sim, "data": data})
            
            for r in sorted(res, key=lambda x: x['sim'], reverse=True)[:5]:
                with st.expander(f"📌 {r['name']} - Giống {round(r['sim'],1)}%"):
                    col_a, col_b = st.columns(2)
                    col_a.image(r['data']['img'], width=250)
                    col_b.write("So sánh thông số:")
                    col_b.json(r['data']['spec'])
