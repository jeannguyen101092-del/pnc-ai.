# ✅ AI FASHION AUDITOR V37.2 - TỐI ƯU CHO MẪU EXPRESS
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

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V37.2", page_icon="📊")

# ================= AI MODELS =================
@st.cache_resource
def load_models():
    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feat_ext = torch.nn.Sequential(*(list(base.children())[:-1])).eval()
    classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    return feat_ext, classifier

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
            feat = model_feat(img_t).flatten().numpy().tolist()
            preds = model_class(img_t)
            cat_id = int(torch.argmax(preds, dim=1))
        return feat, cat_id
    except: return None, 0

# ================= TRÍCH XUẤT NÂNG CAO =================
def extract_techpack(pdf_file):
    data = {"img": None, "tables": pd.DataFrame()}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Tìm ảnh Sketch
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH","FRONT","STYLE","EXPRESS"]):
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
                    if len(df.columns) >= 3: all_dfs.append(df)
            
            if all_dfs:
                # Ưu tiên bảng có nhiều cột (thường là bảng thông số size)
                main_df = max(all_dfs, key=lambda x: len(x.columns))
                # Tự động tìm dòng tiêu đề
                for idx, row in main_df.iterrows():
                    row_s = [str(x).upper() for x in row if x]
                    # Tìm dòng chứa Description hoặc các ký hiệu size phổ biến
                    if any(k in " ".join(row_s) for k in ["DESC", "POM", "TOL", "MEASURE", "SIZE"]):
                        main_df.columns = [str(c).replace('\n',' ').strip().upper() for c in row]
                        data["tables"] = main_df.iloc[idx+1:].reset_index(drop=True)
                        break
                # Nếu vẫn rỗng, lấy đại bảng lớn nhất
                if data["tables"].empty:
                    data["tables"] = main_df
        return data
    except: return None

# ================= MAIN APP =================
supabase = create_client(URL, KEY)
samples = []
try:
    res = supabase.table("ai_data").select("file_name, category_id, vector, spec_json").execute()
    samples = res.data if res.data else []
except:
    res = supabase.table("ai_data").select("file_name, vector, spec_json").execute()
    samples = res.data if res.data else []

with st.sidebar:
    st.header("📂 Kho dữ liệu")
    st.metric("Số mẫu trong kho", len(samples))
    files = st.file_uploader("Nạp mẫu gốc (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Bắt đầu nạp"):
        p_text = st.empty()
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                vec, cat = get_ai_data(d['img'])
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png","x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    # Lưu bảng chuẩn JSON
                    specs = d['tables'].to_json(orient='records')
                    supabase.table("ai_data").upsert({
                        "file_name": name, "vector": vec, "category_id": cat, 
                        "spec_json": specs, "image_url": url
                    }).execute()
                except: pass
            pct = (i + 1) / len(files)
            p_bar.progress(pct)
            p_text.text(f"Đang nạp: {int(pct*100)}%")
        st.success("Đã nạp xong!"); st.rerun()

st.title("🔍 AI FASHION AUDITOR V37.2")
test_file = st.file_uploader("Upload PDF cần kiểm tra", type=["pdf"])

if test_file:
    test = extract_techpack(test_file)
    if test and test['img']:
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(test['img'], caption="Bản vẽ phát hiện được")
            t_vec, t_cat = get_ai_data(test['img'])

        if samples and t_vec:
            # Lọc cùng Category (Quần/Áo)
            valid_samples = [s for s in samples if s.get('category_id') == t_cat]
            if not valid_samples: valid_samples = samples 
            
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
                    st.subheader(f"Mẫu khớp: {ref['file_name']}")
                    st.write(f"Độ tương đồng AI: **{sim_best*100:.1f}%**")
                    st.progress(sim_best)

                    df_t = test['tables']
                    if not df_t.empty:
                        # --- LỌC SIZE THÔNG MINH ---
                        # Bỏ qua các cột thông tin rác
                        noise = ['DESC', 'POM', 'NO', 'TOL', 'ITEM', 'UNNAMED', 'INDEX', 'STT', 'METHOD', 'COMMENTS', 'SAMPLE']
                        size_cols = [c for c in df_t.columns if str(c).strip() and not any(n in str(c).upper() for n in noise)]
                        # Ưu tiên các cột có tên ngắn (S, M, 30, 32...)
                        size_cols = sorted(size_cols, key=len)
                        
                        sel_size = st.selectbox("🎯 Chọn Size đối soát (trong bảng):", size_cols)

                        try:
                            df_ref = pd.read_json(io.StringIO(ref['spec_json']))
                            audit = []
                            # Tìm cột Description (Hạng mục)
                            desc_col = next((c for c in df_t.columns if "DESC" in str(c).upper() or "POM" in str(c).upper() or "ITEM" in str(c).upper()), df_t.columns[0])
                            
                            for _, row in df_t.iterrows():
                                h_muc = str(row[desc_col]).strip()
                                if not h_muc or h_muc.upper() in ['NAN', 'NONE', '']: continue
                                
                                # Khớp dòng Description (Lấy 6 ký tự đầu để tăng độ khớp)
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.upper().str.contains(h_muc[:6].upper(), na=False)]
                                if not match.empty:
                                    try:
                                        v1_raw = re.findall(r"\d+\.?\d*", str(row[sel_size]))
                                        v2_raw = re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))
                                        if v1_raw and v2_raw:
                                            v1, v2 = float(v1_raw[0]), float(v2_raw[0])
                                            diff = round(v1 - v2, 2)
                                            audit.append({
                                                "Hạng mục": h_muc, 
                                                "Thực tế": v1, 
                                                "Mẫu gốc": v2, 
                                                "Lệch": diff, 
                                                "Kquả": "✅ OK" if abs(diff)<=0.5 else "❌ LỆCH"
                                            })
                                    except: continue
                            if audit:
                                st.table(pd.DataFrame(audit))
                                output = io.BytesIO()
                                pd.DataFrame(audit).to_excel(output, index=False)
                                st.download_button("📥 Xuất báo cáo Excel", output.getvalue(), f"Audit_{ref['file_name']}.xlsx")
                            else:
                                st.warning("⚠️ Không tìm thấy hạng mục khớp nhau. Vui lòng chọn lại cột Size hoặc kiểm tra Description.")
                        except: st.error("Lỗi: Mẫu gốc trong kho có cấu trúc bảng không tương thích.")
                    else:
                        st.error("Không tìm thấy bảng thông số kỹ thuật trong PDF.")
