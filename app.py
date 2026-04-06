import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, time, logging
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from datetime import datetime

# =================================================================
# 1. CẤU HÌNH HỆ THỐNG & CSS TÙY CHỈNH
# =================================================================
st.set_page_config(layout="wide", page_title="AI Fashion Pro V12.0", page_icon="👔")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e6ed; }
    .stDataFrame { border-radius: 10px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# Cấu hình Supabase (Thay thông tin của bạn)
SUPABASE_URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
SUPABASE_KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

# =================================================================
# 2. LỚP QUẢN LÝ DỮ LIỆU (SUPABASE MANAGER)
# =================================================================
class DataManager:
    def __init__(self):
        try:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"❌ Lỗi kết nối Database: {e}")

    def upload_image(self, path, content):
        try:
            self.client.storage.from_(BUCKET_NAME).upload(
                path, content, {"content-type": "image/webp", "x-upsert": "true"}
            )
            return self.client.storage.from_(BUCKET_NAME).get_public_url(path)
        except:
            return self.client.storage.from_(BUCKET_NAME).get_public_url(path)

    def upsert_sample(self, data):
        return self.client.table("ai_data").upsert(data, on_conflict="file_name").execute()

    def get_all_samples(self):
        res = self.client.table("ai_data").select("*").execute()
        return res.data if res else []

# =================================================================
# 3. LỚP TRÍ TUỆ NHÂN TẠO (AI ENGINE)
# =================================================================
class AIEngine:
    def __init__(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(
            *(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])
        ).eval()
        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_vector(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            return self.feature_extractor(tensor).flatten().numpy()

# =================================================================
# 4. LỚP TRÍCH XUẤT DỮ LIỆU (FASHION PARSER)
# =================================================================
class FashionParser:
    @staticmethod
    def parse_number(text):
        try:
            t = str(text).strip().replace('-', ' ')
            nums = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t)
            if not nums: return 0
            val = nums[0]
            if ' ' in val:
                parts = val.split()
                return float(parts[0]) + eval(parts[1])
            return eval(val) if '/' in val else float(val)
        except: return 0

    @staticmethod
    def excel_to_img(file_obj):
        try:
            df = pd.read_excel(file_obj).dropna(how='all', axis=0).fillna("")
            fig, ax = plt.subplots(figsize=(18, len(df.head(50)) * 0.5 + 2))
            ax.axis('off')
            ax.table(cellText=df.head(50).values, colLabels=df.columns, loc='center', cellLoc='left').set_fontsize(12)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            return buf.getvalue()
        except: return None

    def extract_pdf(self, pdf_path):
        specs, text, base_size = {}, "", "8"
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text() or ""
                    text += content
                    # Tìm Base Size chuẩn hơn
                    m = re.search(r'(?:Base|Sample|Ref)\s*Size\s*[:\s]\s*(\w+)', content, re.I)
                    if m: base_size = m.group(1).upper()

                    for table in page.extract_tables():
                        if not table or len(table) < 2: continue
                        # Tìm hàng Header
                        h_idx, header = -1, []
                        for i, row in enumerate(table[:10]):
                            row_up = [str(x or "").strip().upper() for x in row]
                            if any(base_size in x for x in row_up):
                                h_idx, header = i, row_up; break
                        
                        if h_idx != -1:
                            # Tìm cột Size mục tiêu
                            b_idx = next((i for i, v in enumerate(header) if base_size in v and len(v) < 6), -1)
                            if b_idx != -1:
                                for r in table[h_idx + 1:]:
                                    desc = " ".join([str(x or "") for x in r[:b_idx]]).strip().upper()
                                    val = self.parse_number(r[b_idx])
                                    if val > 0 and len(desc) > 5:
                                        specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:120]] = round(float(val), 3)
            
            doc = fitz.open(pdf_path)
            img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
            doc.close()
            return {"spec": specs, "img": img, "text": text}
        except: return None

# =================================================================
# 5. GIAO DIỆN CHÍNH (STREAMLIT APP)
# =================================================================
db = DataManager()
ai = AIEngine()
parser = FashionParser()

# Khởi tạo Session State
if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ AI FASHION PRO")
    samples = db.get_all_samples()
    st.metric("📦 Kho lưu trữ", f"{len(samples)} mẫu")
    
    st.subheader("🎯 Chọn mẫu đối chiếu")
    list_names = [s['file_name'] for s in samples]
    sel = st.selectbox("Mã hàng cố định", ["-- Chọn --"] + list_names)
    if sel != "-- Chọn --":
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    st.subheader("🚀 Nạp dữ liệu mới")
    up_files = st.file_uploader("PDF & Excel", accept_multiple_files=True, type=['pdf','xlsx','xls'], key=st.session_state.up_key)
    
    if up_files and st.button("BẮT ĐẦU NẠP"):
        groups = {}
        for f in up_files:
            nums = re.findall(r'\d+', f.name)
            ma = max(nums, key=len) if nums else "UNKNOWN"
            ext = "." + f.name.split('.')[-1].lower()
            if ma not in groups: groups[ma] = {}
            groups[ma][ext] = f
        
        for ma, p in groups.items():
            if p.get('.pdf') and (p.get('.xlsx') or p.get('.xls')):
                with st.spinner(f"Đang nạp {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(p['.pdf'].getbuffer())
                    data = parser.extract_pdf("tmp.pdf")
                    exl_img = parser.excel_to_img(p.get('.xlsx') or p.get('.xls'))
                    if data and exl_img:
                        vec = ai.generate_vector(data['img']).tolist()
                        url_t = db.upload_image(f"{ma}_t.webp", data['img'])
                        url_e = db.upload_image(f"{ma}_e.webp", exl_img)
                        db.upsert_sample({
                            "file_name": ma, "vector": vec, "spec_json": data['spec'],
                            "img_url": url_t, "excel_img_url": url_e
                        })
                        st.toast(f"✅ Xong mã {ma}")
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN PANEL ---
st.header("👔 Hệ Thống So Sánh Thông Số Tự Động")
test_pdf = st.file_uploader("1. Tải PDF cần kiểm tra", type="pdf")

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    test_data = parser.extract_pdf("test.pdf")
    
    if test_data:
        # Tự động tìm mã giống nhất bằng AI
        if st.button("🤖 AI: TỰ ĐỘNG TÌM MẪU TƯƠNG ĐỒNG"):
            test_vec = ai.generate_vector(test_data['img'])
            best_sim, best_s = -1, None
            for s in samples:
                if s.get('vector'):
                    sim = cosine_similarity([test_vec], [np.array(s['vector'])])[0][0]
                    if sim > best_sim: best_sim, best_s = sim, s
            st.session_state.target = best_s
            st.session_state.sim_score = round(best_sim * 100, 1)

        # Hiển thị 3 cột
        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            st.image(test_data['img'], caption="🖼️ MẪU ĐANG KIỂM TRA", use_container_width=True)
        
        with c2:
            target = st.session_state.target
            if target:
                sim = st.session_state.get('sim_score', '---')
                st.image(target['img_url'], caption=f"📁 MẪU KHO ({target['file_name']}) - GIỐNG {sim}%", use_container_width=True)
                st.divider()
                st.image(target['excel_img_url'], caption="📊 ĐỊNH MỨC GỐC", use_container_width=True)
            else:
                st.info("👈 Vui lòng chọn mẫu ở Sidebar")

        with c3:
            if target:
                st.subheader(f"📊 Kết quả đối chiếu: {target['file_name']}")
                results = []
                for k_t, v_t in test_data['spec'].items():
                    # Fuzzy matching tên thông số
                    best_m, high_r = None, 0
                    for k_db in target['spec_json'].keys():
                        r = SequenceMatcher(None, k_t, k_db).ratio()
                        if r > high_r: high_r, best_m = r, k_db
                    
                    v_db = target['spec_json'].get(best_m, 0) if high_r > 0.6 else "N/A"
                    diff = round(v_t - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
                    results.append({
                        "Thông số": k_t, "Test": v_t, "Kho": v_db, 
                        "Chênh lệch": diff, "Kết quả": "✅ OK" if diff == 0 else "❌ SAI"
                    })
                
                df = pd.DataFrame(results)
                st.dataframe(df.style.map(
                    lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''),
                    subset=['Kết quả']
                ), use_container_width=True, height=600)
                
                # Xuất file
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr: df.to_excel(wr, index=False)
                st.download_button("📥 XUẤT EXCEL SO SÁNH", out.getvalue(), f"Result_{target['file_name']}.xlsx")
