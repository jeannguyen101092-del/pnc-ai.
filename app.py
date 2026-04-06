import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, time, datetime, logging
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import xlsxwriter

# =================================================================
# 1. HỆ THỐNG CẤU HÌNH & TRẠNG THÁI (CORE CONFIG)
# =================================================================
ST_PAGE_TITLE = "AI FASHION PRO - ENTERPRISE V12.5"
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title=ST_PAGE_TITLE, page_icon="🛡️")

# Custom CSS cho giao diện chuyên nghiệp
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { border: 1px solid #d1d5db; padding: 15px; border-radius: 10px; background: white; }
    .status-ok { color: #059669; font-weight: bold; }
    .status-fail { color: #dc2626; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1f2937; color: white; }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# 2. LỚP QUẢN LÝ AI (AI VISION ENGINE)
# =================================================================
class AIVisionEngine:
    def __init__(self):
        # Sử dụng MobileNetV2 làm backbone để trích xuất đặc trưng hình dáng
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(
            *(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])
        ).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_image_embedding(self, img_bytes):
        """Chuyển đổi hình ảnh thành Vector 1280 chiều để so sánh kiểu dáng"""
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = self.encoder(tensor).flatten().numpy()
            return embedding
        except Exception as e:
            logging.error(f"AI Error: {e}")
            return None

# =================================================================
# 3. LỚP XỬ LÝ DỮ LIỆU PDF/EXCEL (FASHION DATA PARSER)
# =================================================================
class FashionParser:
    @staticmethod
    def clean_text(text):
        return re.sub(r'[^A-Z0-9\s/]', '', str(text).upper()).strip()

    @staticmethod
    def fraction_to_float(text):
        """Chuyển đổi phân số ngành may (VD: 15 1/2) sang số thập phân (15.5)"""
        try:
            text = str(text).strip().replace('-', ' ')
            if not text: return 0
            matches = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', text)
            if not matches: return 0
            val = matches[0]
            if ' ' in val:
                p = val.split()
                return float(p[0]) + eval(p[1])
            return eval(val) if '/' in val else float(val)
        except: return 0

    def extract_specs_from_pdf(self, pdf_path):
        """Hệ thống quét bảng PDF đa tầng, không bỏ sót trang nào"""
        specs = {}
        all_text = ""
        base_size = "8"
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    all_text += txt
                    
                    # Tìm Base Size trong toàn văn bản
                    size_match = re.search(r'(?:Base|Sample|Reference)\s*Size\s*[:\s]\s*(\w+[\-?]?)', txt, re.I)
                    if size_match: base_size = size_match.group(1).upper()

                    tables = page.extract_tables()
                    for table in tables:
                        if not table or len(table) < 2: continue
                        
                        # Soi Header để tìm cột số đo
                        h_idx, header = -1, []
                        for i, row in enumerate(table[:10]):
                            row_clean = [str(x or "").strip().upper() for x in row]
                            if any(base_size == x or (base_size in x and len(x) < 5) for x in row_clean):
                                h_idx, header = i, row_clean
                                break
                        
                        if h_idx != -1:
                            # Xác định cột Description (thường nằm trước cột số đo)
                            b_idx = next((idx for idx, v in enumerate(header) if base_size in v and len(v) < 6), -1)
                            if b_idx != -1:
                                for r in table[h_idx + 1:]:
                                    if not r or len(r) <= b_idx: continue
                                    # Gộp các cột mô tả lại để trích xuất đầy đủ tên điểm đo
                                    desc_raw = " ".join([str(x or "") for x in r[:b_idx]]).strip()
                                    desc_clean = self.clean_text(desc_raw)
                                    val = self.fraction_to_float(r[b_idx])
                                    if val > 0.1 and len(desc_clean) > 5:
                                        specs[desc_clean[:120]] = round(float(val), 3)

            # Chụp ảnh Preview trang 1 chất lượng cao
            doc = fitz.open(pdf_path)
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            doc.close()
            
            return {"spec": specs, "img": img_bytes, "full_text": all_text}
        except Exception as e:
            st.error(f"Lỗi Parser: {e}")
            return None

    def excel_to_image(self, file_obj):
        """Chuyển đổi bảng Excel thành ảnh trực quan để đối chiếu thủ công khi cần"""
        try:
            # Hỗ trợ cả định dạng cũ .xls
            engine = 'xlrd' if file_obj.name.lower().endswith('.xls') else 'openpyxl'
            df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
            
            fig, ax = plt.subplots(figsize=(20, len(df.head(60)) * 0.6 + 2))
            ax.axis('off')
            ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1, 2.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=160)
            plt.close(fig)
            return buf.getvalue()
        except Exception as e:
            st.error(f"Lỗi Excel: {e}")
            return None

# =================================================================
# 4. QUẢN TRỊ GIAO DIỆN & LOGIC NGHIỆP VỤ (BUSINESS LOGIC)
# =================================================================
class AppController:
    def __init__(self):
        self.db: Client = create_client(URL, KEY)
        self.ai = AIVisionEngine()
        self.parser = FashionParser()
        self.init_session()

    def init_session(self):
        if "target" not in st.session_state: st.session_state.target = None
        if "sim_score" not in st.session_state: st.session_state.sim_score = 0
        if "up_key" not in st.session_state: st.session_state.up_key = 100

    def run_sidebar(self):
        with st.sidebar:
            st.title("🛡️ CONTROL PANEL")
            # Hiển thị Metric
            res = self.db.table("ai_data").select("file_name").execute()
            all_samples = res.data if res else []
            st.metric("📦 TỔNG MẪU TRONG KHO", len(all_samples))

            # Bộ lọc chọn mã cố định
            list_ma = [s['file_name'] for s in all_samples]
            sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU CỐ ĐỊNH", ["-- Click để chọn --"] + list_ma)
            if sel != "-- Click để chọn --":
                full_data = self.db.table("ai_data").select("*").eq("file_name", sel).execute()
                if full_data.data:
                    st.session_state.target = full_data.data[0]
                    st.session_state.sim_score = 100

            st.divider()
            # Khu vực nạp dữ liệu mới
            st.subheader("🚀 NẠP DỮ LIỆU MỚI")
            files = st.file_uploader("Kéo thả PDF & Excel", accept_multiple_files=True, 
                                    type=['pdf', 'xlsx', 'xls'], key=st.session_state.up_key)
            
            if files and st.button("BẮT ĐẦU XỬ LÝ & LƯU KHO"):
                self.process_uploads(files)

    def process_uploads(self, files):
        groups = {}
        for f in files:
            # Tìm mã số thông minh (dãy số dài nhất trong tên file)
            nums = re.findall(r'\d+', f.name)
            if nums:
                ma = max(nums, key=len)
                ext = "." + f.name.split('.')[-1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f
        
        for ma, parts in groups.items():
            f_pdf = parts.get('.pdf')
            f_exl = parts.get('.xlsx') or parts.get('.xls')
            
            if f_pdf and f_exl:
                with st.spinner(f"Đang phân tích mã {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_pdf.getbuffer())
                    pdf_data = self.parser.extract_specs_from_pdf("tmp.pdf")
                    exl_img = self.parser.excel_to_image(f_exl)
                    
                    if pdf_data and exl_img:
                        # AI Vectorizing
                        vec = self.ai.get_image_embedding(pdf_data['img']).tolist()
                        
                        # Storage Upload
                        try:
                            self.db.storage.from_(BUCKET).upload(f"{ma}_t.webp", pdf_data['img'], {"x-upsert": "true"})
                            self.db.storage.from_(BUCKET).upload(f"{ma}_e.webp", exl_img, {"x-upsert": "true"})
                            
                            url_t = self.db.storage.from_(BUCKET).get_public_url(f"{ma}_t.webp")
                            url_e = self.db.storage.from_(BUCKET).get_public_url(f"{ma}_e.webp")
                            
                            # Upsert DB
                            self.db.table("ai_data").upsert({
                                "file_name": ma, "vector": vec, "spec_json": pdf_data['spec'],
                                "img_url": url_t, "excel_img_url": url_e, "updated_at": str(datetime.datetime.now())
                            }).execute()
                            st.toast(f"✅ Mã {ma} đã sẵn sàng!")
                        except Exception as e:
                            st.error(f"Lỗi nạp mã {ma}: {e}")
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.session_state.up_key += 1
        st.rerun()

    def run_main(self):
        st.title("👔 HỆ THỐNG SO SÁNH THÔNG SỐ TỰ ĐỘNG")
        test_pdf = st.file_uploader("1. TẢI FILE PDF CẦN KIỂM TRA", type="pdf")
        
        if test_pdf:
            with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
            test_data = self.parser.extract_specs_from_pdf("test.pdf")
            
            if test_data:
                # Nút AI Search
                if st.button("🤖 AI: TỰ ĐỘNG NHẬN DIỆN MÃ TƯƠNG ĐỒNG (THÔNG MINH)"):
                    self.search_similar(test_data['img'])

                # Hiển thị 3 cột chính
                c1, c2, c3 = st.columns([1, 1, 1.6])
                
                with c1:
                    st.subheader("🖼️ PDF ĐANG TEST")
                    st.image(test_data['img'], use_container_width=True)
                
                with c2:
                    target = st.session_state.target
                    st.subheader("📁 DỮ LIỆU ĐỐI CHIẾU")
                    if target:
                        st.image(target['img_url'], caption=f"Mã kho: {target['file_name']} (Giống {st.session_state.sim_score}%)", use_container_width=True)
                        st.divider()
                        st.image(target['excel_img_url'], caption="Bảng định mức Excel gốc", use_container_width=True)
                    else:
                        st.info("👈 Hãy chọn mã ở cột bên trái hoặc dùng AI tìm kiếm.")

                with c3:
                    st.subheader("📊 BẢNG ĐỐI CHIẾU CHI TIẾT")
                    if target:
                        self.render_comparison_table(test_data['spec'], target)

    def search_similar(self, test_img_bytes):
        """Tìm mã hàng giống nhất trong kho bằng AI Vector"""
        test_vec = self.ai.get_image_embedding(test_img_bytes)
        res = self.db.table("ai_data").select("*").execute()
        samples = res.data if res else []
        
        best_sim, best_item = -1, None
        for s in samples:
            if s.get('vector'):
                sim = cosine_similarity([test_vec], [np.array(s['vector'])])[0][0]
                if sim > best_sim:
                    best_sim, best_item = sim, s
        
        if best_item:
            st.session_state.target = best_item
            st.session_state.sim_score = round(best_sim * 100, 1)
            st.rerun()

    def render_comparison_table(self, test_specs, target):
        """Xử lý logic so sánh và hiển thị bảng kết quả"""
        rows = []
        db_specs = target['spec_json']
        
        for k_test, v_test in test_specs.items():
            # Thuật toán so sánh chuỗi mờ (Fuzzy String Matching)
            best_match, high_ratio = None, 0
            for k_db in db_specs.keys():
                ratio = SequenceMatcher(None, k_test, k_db).ratio()
                if ratio > high_ratio: high_ratio, best_match = ratio, k_db
            
            v_db = db_specs.get(best_match, 0) if high_ratio > 0.65 else "N/A"
            diff = round(v_test - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
            
            # Đánh giá trạng thái
            if diff == "N/A": status = "❓ MỚI"
            elif abs(diff) <= 0.125: status = "✅ OK" # Ngưỡng sai số 1/8 inch
            else: status = "❌ SAI"
            
            rows.append({
                "Điểm đo (Description)": k_test,
                "Số đo PDF Test": v_test,
                "Số đo Kho Gốc": v_db,
                "Chênh lệch": diff,
                "Kết quả": status
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            # Hiển thị DataFrame với Style
            st.dataframe(df.style.map(
                lambda x: 'background-color: #fee2e2; color: #b91c1c' if x == "❌ SAI" else 
                          ('background-color: #d1fae5; color: #047857' if x == "✅ OK" else ''),
                subset=['Kết quả']
            ), use_container_width=True, height=600)
            
            # Xuất Excel báo cáo
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='xlsxwriter')
            st.download_button("📥 TẢI BÁO CÁO SO SÁNH (EXCEL)", output.getvalue(), 
                               f"Report_{target['file_name']}.xlsx", "application/vnd.ms-excel")

# =================================================================
# 6. KHỞI CHẠY HỆ THỐNG
# =================================================================
if __name__ == "__main__":
    app = AppController()
    app.run_sidebar()
    app.run_main()
