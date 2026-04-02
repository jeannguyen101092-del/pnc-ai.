import streamlit as st
import os, io, pickle, torch, pdfplumber, re, fitz
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# --- 1. KẾT NỐI GOOGLE DRIVE ---
# Lấy ID thư mục từ link bạn gửi
FOLDER_ID = '1P9EL2-BC0du_im533bsv1KSzr6vsvKr-'
DB_PATH = 'database_ai.pkl'

# Lấy "chìa khóa" JSON từ Secrets bạn đã dán
info = st.secrets["gdrive_service_account"]
creds = service_account.Credentials.from_service_account_info(info)
drive_service = build('drive', 'v3', credentials=creds)

st.set_page_config(layout="wide", page_title="AI MASTER PRO - DRIVE CLOUD", page_icon="🛡️")

# --- 2. BỘ NÃO AI (CHẠY TRÊN CPU CLOUD) ---
@st.cache_resource
def load_ai():
    model = models.resnet50(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# (Các hàm parse_val và KEY_GROUPS giữ nguyên như bản cũ của bạn)
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
        if ' ' in t: p = t.split(); return float(p[0]) + eval(p[1])
        if '/' in t: return eval(t)
        return float(t)
    except: return None

def get_data(pdf_content):
    specs = {}; category = "UNKNOWN"
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages]).lower()
            if any(x in txt for x in ["jacket", "shirt", "top", "hoodie", "tee"]): category = "TOP"
            elif any(x in txt for x in ["pant", "jean", "short", "bottom"]): category = "BOTTOM"
            
            all_rows = []
            for p in pdf.pages:
                if p.extract_table(): all_rows.extend(p.extract_table())
            
            # Logic tìm Sample Size và Specs tương tự bản cũ
            for r in all_rows:
                r_c = [str(x).strip() for x in r if x]; r_s = " ".join(r_c)
                for std, keys in KEY_GROUPS.items():
                    if any(k.lower() in r_s.lower() for k in keys):
                        val = parse_val(r_c[-1]) if r_c else None # Lấy cột cuối làm mặc định
                        if val and val > 4: specs[std] = val

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img = Image.open(io.BytesIO(doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")))
        w,h = img.size; crop = img.crop((int(w*0.05), int(h*0.15), int(w*0.58), int(h*0.85)))
        return {"spec": specs, "img": crop, "cat": category}
    except: return None

# --- 3. ĐỒNG BỘ DRIVE & QUÉT FILE MỚI ---
def sync_drive():
    db = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f: db = pickle.load(f)
    
    # Tìm tất cả file PDF trong thư mục Drive
    results = drive_service.files().list(q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf'",
                                         fields="files(id, name)").execute()
    items = results.get('files', [])
    
    new_items = [i for i in items if i['name'] not in db]
    
    if new_items:
        st.info(f"🆕 Phát hiện {len(new_items)} file mới trên Drive. Đang quét...")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        for item in new_items:
            # Tải file từ Drive
            request = drive_service.files().get_media(fileId=item['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            
            data = get_data(fh.getvalue())
            if data:
                with torch.no_grad():
                    vec = ai_brain(tf(data['img'].convert('RGB')).unsqueeze(0)).flatten().detach().cpu().numpy()
                db[item['name']] = {"vector": vec, "img": data['img'], "spec": data['spec'], "cat": data['cat']}
        
        with open(DB_PATH, "wb") as f: pickle.dump(db, f)
    return db

# --- 4. GIAO DIỆN CHÍNH ---
st.title("🛡️ AI MASTER PRO - DRIVE CONNECT")
db = sync_drive()
st.sidebar.metric("📦 TỔNG MẪU TRONG KHO", len(db))

up = st.file_uploader("📤 Thả PDF mẫu vào đây", type="pdf")
if up:
    target = get_data(up.read())
    if target:
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            t_vec = ai_brain(tf(target['img'].convert('RGB')).unsqueeze(0)).flatten().detach().cpu().numpy()
        
        c1, c2 = st.columns([1, 2.3])
        with c1:
            st.subheader("🔍 Mẫu Gốc"); st.image(target['img'], use_container_width=True); st.json(target['spec'])
        with c2:
            st.subheader("✅ Top Tương Đồng")
            res = []
            for name, data in db.items():
                sim = float(cosine_similarity(t_vec.reshape(1,-1), data['vector'].reshape(1,-1))) * 100
                if sim > 70:
                    diff = [f"**{k}**: {target['spec'][k]} vs {data['spec'][k]}" for k in set(target['spec']) & set(data['spec'])]
                    res.append({"name": name, "sim": sim, "diff": diff, "img": data['img']})
            
            for r in sorted(res, key=lambda x: x['sim'], reverse=True)[:10]:
                with st.expander(f"📌 {r['name']} - Giống {round(r['sim'],1)}%"):
                    st.image(r['img'], width=300)
                    for line in r['diff']: st.markdown(line)
