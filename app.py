import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL/KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V37", page_icon="📊")

# ================= HỆ THỐNG AI PHÂN LOẠI =================
@st.cache_resource
def load_models():
    # Model lấy đặc trưng (Feature)
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*(list(base_model.children())[:-1])).eval()
    # Model phân loại (Category) để phân biệt Áo/Quần/Váy
    classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    return feature_extractor, classifier

model_feat, model_class = load_models()

def get_ai_data(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        img_t = tf(img).unsqueeze(0)
        with torch.no_grad():
            # 1. Lấy Vector đặc trưng
            feat = model_feat(img_t).flatten().numpy()
            # 2. Lấy Category ID (Để phân biệt loại đồ)
            preds = model_class(img_t)
            category_id = int(torch.argmax(preds, dim=1))
        return feat.tolist(), category_id
    except: return None, None

# ================= TRÍCH XUẤT BẢNG & SIZE =================
def extract_techpack(pdf_file):
    data = {"img": None, "tables": pd.DataFrame()}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Tìm ảnh Sketch (Trang chứa thiết kế)
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH","FRONT","STYLE","DESIGN"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if not data["img"]: data["img"] = doc.load_page(0).get_pixmap().tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_dfs = []
            for page in pdf.pages:
                tbs = page.extract_tables()
                for t in tbs:
                    df = pd.DataFrame(t).dropna(how='all')
                    if len(df.columns) > 3: all_dfs.append(df)
            if all_dfs:
                main_df = max(all_dfs, key=len) # Bảng POM thường dài nhất
                # Tìm header chứa Description/Size
                for idx, row in main_df.iterrows():
                    row_s = [str(x).upper() for x in row if x]
                    if any("DESC" in s or "POM" in s or "SIZE" in s for s in row_s):
                        main_df.columns = [str(c).replace('\n',' ').strip() for c in row]
                        data["tables"] = main_df.iloc[idx+1:].reset_index(drop=True)
                        break
        return data
    except: return None

# ================= GIAO DIỆN CHÍNH =================
supabase = create_client(URL, KEY)
res = supabase.table("ai_data").select("file_name, category_id, vector, spec_json").execute()
samples = res.data if res.data else []

with st.sidebar:
    st.header("📂 Kho dữ liệu")
    st.metric("Số mẫu trong kho", len(samples)) # Hiển thị tổng số file
    
    files = st.file_uploader("Nạp mẫu mới (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Bắt đầu nạp"):
        progress_text = st.empty()
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                vec, cat = get_ai_data(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png","x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    specs = d['tables'].to_json(orient='records')
                    supabase.table("ai_data").upsert({
                        "file_name": name, "vector": vec, "category_id": cat, 
                        "spec_json": specs, "image_url": url
                    }).execute()
                except: pass
            pct = (i + 1) / len(files)
            p_bar.progress(pct)
            progress_text.text(f"Đang nạp: {int(pct*100)}%")
        st.success("Đã nạp xong!"); st.rerun()

st.title("🔍 AI FASHION AUDITOR V37")
test_file = st.file_uploader("Upload PDF cần kiểm tra", type=["pdf"])

if test_file:
    test = extract_techpack(test_file)
    if test and test['img']:
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(test['img'], caption="Bản vẽ phát hiện được")
            t_vec, t_cat = get_ai_data(test['img'])

        if samples and t_vec:
            # 🔥 BỘ LỌC THÔNG MINH: Chỉ so sánh những mẫu cùng Category (Áo vs Áo, Quần vs Quần)
            # ResNet category ID giúp nhận diện loại trang phục
            valid_samples = [s for s in samples if s.get('category_id') == t_cat]
            if not valid_samples: valid_samples = samples # Fallback nếu ko tìm thấy cùng loại
            
            results = []
            for s in valid_samples:
                try:
                    sim = float(cosine_similarity([t_vec], [s['vector']]))
                    results.append((s, sim))
                except: continue
            
            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                with col2:
                    ref, sim_best = results[0]
                    st.subheader(f"Kết quả: {ref['file_name']}")
                    st.write(f"Độ tương đồng: **{sim_best*100:.1f}%**")
                    st.progress(sim_best)

                    df_t = test['tables']
                    if not df_t.empty:
                        # 🎯 CHỈ CHỌN SIZE: Lọc bỏ các cột Description, No, Tol...
                        noise = ['DESC', 'POM', 'NO', 'TOL', 'ITEM', 'UNNAMED', 'INDEX', 'STT', 'METHOD']
                        size_cols = [c for c in df_t.columns if c and not any(n in str(c).upper() for n in noise)]
                        # Ưu tiên các cột có tên ngắn (thường là size số hoặc chữ)
                        size_cols = [c for c in size_cols if len(str(c)) < 10]
                        
                        sel_size = st.selectbox("Chọn cột thông số (Size):", size_cols)

                        try:
                            df_ref = pd.read_json(io.StringIO(ref['spec_json']))
                            audit = []
                            # Tìm cột Description (Hạng mục)
                            desc_col = next((c for c in df_t.columns if "DESC" in str(c).upper() or "POM" in str(c).upper()), df_t.columns[0])
                            
                            for _, row in df_t.iterrows():
                                h_muc = str(row[desc_col]).strip()
                                if not h_muc or h_muc.upper() == 'NAN': continue
                                
                                # Khớp dòng Description
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.contains(h_muc[:8], case=False, na=False)]
                                if not match.empty:
                                    try:
                                        v1 = re.findall(r"\d+\.?\d*", str(row[sel_size]))[0]
                                        v2 = re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))[0]
                                        diff = round(float(v1) - float(v2), 2)
                                        audit.append({"Hạng mục": h_muc, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": diff, "Kquả": "✅" if abs(diff)<=0.5 else "❌"})
                                    except: continue
                            if audit:
                                st.table(pd.DataFrame(audit))
                                # Nút xuất Excel (Tự tạo file)
                                output = io.BytesIO()
                                pd.DataFrame(audit).to_excel(output, index=False)
                                st.download_button("📥 Xuất báo cáo Excel", output.getvalue(), "Audit_Report.xlsx")
                        except: st.error("Lỗi cấu trúc bảng mẫu gốc.")
