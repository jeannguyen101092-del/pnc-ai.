import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"           
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Database!")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V16.2", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 1500

@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

# ================= HỆ THỐNG PHÂN TÍCH =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): 
            # Chuyển đổi sang list để lưu được vào Supabase
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

def analyze_garment_logic(text):
    t = str(text).upper()
    details = []
    if 'CARGO' in t: details.append("📦 Túi Hộp (Cargo Pocket)")
    if 'SLANT' in t: details.append("📐 Túi Xéo (Slant Pocket)")
    if 'SCOOP' in t or 'HAM ECH' in t: details.append("🐸 Túi Hàm Ếch (Scoop)")
    if 'PATCH' in t: details.append("🎨 Túi Đắp (Patch Pocket)")
    if 'ELASTIC' in t: details.append("🧶 Lưng Thun (Elastic Waist)")
    if 'SKORT' in t: details.append("👗 Quần Váy (Skort)")
    if 'LONG SLEEVE' in t: details.append("🧥 Áo Dài Tay")
    if 'SHORT SLEEVE' in t: details.append("👕 Áo Ngắn Tay")
    if 'A-LINE' in t or 'FLARE' in t: details.append("💃 Váy Xòe (Flare)")
    if 'PENCIL' in t or 'TUM' in t: details.append("👗 Váy Túm/Bút Chì")
    return details

def excel_to_img_matrix(file_obj):
    try:
        ext = file_obj.name.split('.')[-1].lower()
        engine = 'xlrd' if ext == 'xls' else 'openpyxl'
        df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(24, len(df.head(80)) * 0.7 + 2))
        ax.axis('off')
        ax.table(cellText=df.head(80).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1.2, 3)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=180); plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.error(f"❌ Lỗi định dạng Excel: {e}")
        return None

def extract_pdf_ultimate(pdf_path):
    specs, text, base_size = {}, "", "8"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text += txt
                m = re.search(r'(?:Base|Sample|Ref)\s*Size\s*[:\s]\s*(\w+)', txt, re.I)
                if m: base_size = m.group(1).upper()
                for tb in page.extract_tables():
                    if not tb or len(tb) < 2: continue
                    h_idx, header = -1, []
                    for i, row in enumerate(tb[:10]):
                        row_up = [str(x or "").strip().upper() for x in row]
                        if any(base_size in x and len(x) < 6 for x in row_up):
                            h_idx, header = i, row_up; break
                    if h_idx != -1:
                        b_idx = next((idx for idx, v in enumerate(header) if base_size in v and len(v) < 6), -1)
                        if b_idx != -1:
                            for r in tb[h_idx + 1:]:
                                if not r or len(r) <= b_idx: continue
                                desc = " ".join([str(x or "") for x in r[:b_idx]]).strip().upper()
                                val = str(r[b_idx]).strip()
                                if val and len(desc) > 5:
                                    specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:120]] = val
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU V16.2")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    list_ma = [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU CỐ ĐỊNH", ["-- Click chọn --"] + list_ma)
    if sel != "-- Click chọn --": 
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    up_files = st.file_uploader("Nạp PDF & Excel mới", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'], key=st.session_state.up_key)
    
    if up_files and st.button("🚀 NẠP & PHÂN TÍCH CHI TIẾT AI"):
        pdfs = [f for f in up_files if f.name.lower().endswith('.pdf')]
        exls = [f for f in up_files if f.name.lower().endswith(('.xls', '.xlsx'))]
        
        for f_p in pdfs:
            nums_p = set(re.findall(r'\d{3,}', f_p.name)) - {str(x) for x in range(2023, 2027)}
            f_e, ma = None, "UNK"
            for ex in exls:
                nums_e = set(re.findall(r'\d{3,}', ex.name)) - {str(x) for x in range(2023, 2027)}
                common = nums_p.intersection(nums_e)
                if common: 
                    f_e, ma = ex, str(list(common)[0]) # Lấy mã đầu tiên và ép kiểu string
                    break
            
            if f_p and f_e and ma != "UNK":
                with st.spinner(f"AI đang 'soi' mã {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                    d, ex_img = extract_pdf_ultimate("tmp.pdf"), excel_to_img_matrix(f_e)
                    if d and ex_img:
                        vec = get_vector(d['img'])
                        details = analyze_garment_logic(d) # FIX: Truyền text
                        try:
                            # Upload ảnh spec và ảnh excel
                            supabase.storage.from_(BUCKET).upload(f"{ma}_t.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                            supabase.storage.from_(BUCKET).upload(f"{ma}_e.png", ex_img, {"x-upsert": "true", "content-type": "image/png"})
                            
                            u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.png")
                            u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.png")
                            
                            # Lưu vào Database
                            supabase.table("ai_data").upsert({
                                "file_name": ma, 
                                "vector": vec, 
                                "spec_json": d['spec'], 
                                "img_url": u_t, 
                                "excel_img_url": u_e, 
                                "details": details
                            }).execute()
                            st.toast(f"✅ Đã nạp thành công mã {ma}")
                        except Exception as e: st.error(f"Lỗi DB: {e}")
        st.session_state.up_key += 1; st.rerun()

# ================= MAIN UI =================
st.title("🛡️ AI FASHION PRO - SO SÁNH SIÊU CẤP")
test_pdf = st.file_uploader("1. Tải PDF Test (File cần kiểm tra)", type="pdf")
target = st.session_state.target

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    data_test = extract_pdf_ultimate("test.pdf")
    
    if data_test:
        col1, col2 = st.columns(2)
        with col1: st.image(data_test['img'], caption="FILE TEST")
        
        if st.button("🤖 AI: TỰ ĐỘNG NHẬN DIỆN MÃ TƯƠNG ĐỒNG"):
            vec_test = get_vector(data_test['img'])
            best_score, best_match = 0, None
            for s in samples:
                score = cosine_similarity([vec_test], [s['vector']])[0][0]
                if score > best_score:
                    best_score, best_match = score, s
            if best_match:
                st.session_state.target = best_match
                st.success(f"Đã tìm thấy mã {best_match['file_name']} (Giống {best_score:.1%})")
                st.rerun()

        if target:
            with col2: st.image(target['img_url'], caption=f"KHO GỐC: {target['file_name']}")
            
            # --- SO SÁNH THÔNG SỐ VÀ XUẤT FILE ---
            st.divider()
            st.subheader("📏 BẢNG ĐỐI CHIẾU THÔNG SỐ")
            
            comparison = []
            for item, val in data_test['spec'].items():
                val_goc = target['spec_json'].get(item, "---")
                status = "✅ Khớp" if str(val) == str(val_goc) else "❌ Lệch"
                comparison.append({"Hạng mục": item, "Giá trị Test": val, "Giá trị Gốc": val_goc, "Kết quả": status})
            
            df_compare = pd.DataFrame(comparison)
            st.table(df_compare)

            # NÚT XUẤT FILE
            st.write("### 📤 XUẤT BÁO CÁO")
            csv = df_compare.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 TẢI FILE SO SÁNH (.CSV)",
                data=csv,
                file_name=f"SoSanh_{target['file_name']}.csv",
                mime='text/csv',
            )

if not target:
    st.info("💡 Hãy chọn một mã đối chiếu ở Sidebar hoặc bấm 'Tự động nhận diện' sau khi up file Test.")
