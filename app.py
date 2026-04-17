import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'sync_results' not in st.session_state: st.session_state['sync_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower()
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page"]): return 0
        t = t.replace(',', '.')
        # Xử lý số hỗn hợp (Ví dụ: 16 1/4)
        mixed = re.match(r'(\d+)[-\s]+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2)) / int(mixed.group(3))
        # Xử lý phân số (1/2)
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1)) / int(frac.group(2))
        # Số thực bình thường
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if val < 500 else 0
        return 0
    except: return 0

# ================= 3. SCRAPER (OPTIMIZED) =================
def extract_data(file_content, scan_all=False):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        buf = io.BytesIO(); Image.open(io.BytesIO(pix.tobytes("png"))).save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue(); doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            # Audit Mode: Chỉ trang 1 | Version Control: Tất cả trang
            pages = pdf.pages if scan_all else [pdf.pages[0]]
            for page in pages:
                df_w = pd.DataFrame(page.extract_words())
                if df_w.empty: continue
                df_w['y_grid'] = (df_w['top'] / 2).round() * 2
                page_width = page.width
                for y, group in df_w.groupby('y_grid'):
                    sorted_g = group.sort_values('x0')
                    line_txt = " ".join(sorted_g['text'])
                    if any(kw in line_txt.upper() for kw in POM_KWS):
                        numeric_part = sorted_g[sorted_g['x0'] > (page_width * 0.55)]
                        text_part = sorted_g[sorted_g['x0'] <= (page_width * 0.55)]
                        pom_name = re.sub(r'^[A-Z0-9-]+\s+', '', " ".join(text_part['text']).strip())
                        if len(pom_name) > 3:
                            vals = [parse_val(t) for t in numeric_part['text'] if parse_val(t) > 0]
                            for i, v in enumerate(vals):
                                s_key = f"Size_{i+1}"
                                if s_key not in all_specs: all_specs[s_key] = {}
                                all_specs[s_key][pom_name] = v
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparison')
    return output.getvalue()

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        res = supabase.table("ai_data").select("id", count="exact").execute()
        count = res.count or 0
    except: count = 0
    st.metric("Models in Repo", f"{count} SKUs")
    st.divider()
    
    files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"s_{st.session_state['up_key']}")
    # Tìm đến đoạn nạp dữ liệu trong Sidebar và thay bằng logic làm sạch này:

if files and st.button("🚀 SYNCHRONIZE & REPAIR", use_container_width=True):
    logs = []
    for f in files:
        try:
            fb = f.getvalue()
            data = extract_data(fb, scan_all=False)
            
            if data and data['img']:
                # LÀM SẠCH ID: Loại bỏ ký tự đặc biệt (khoảng trắng, ngoặc đơn...)
                clean_name = re.sub(r'[^a-zA-Z0-9]', '_', f.name)
                f_hash = hashlib.md5(fb).hexdigest()
                unique_id = f"{clean_name}_{f_hash[:6]}"
                
                # KIỂM TRA VECTOR: Đảm bảo vector hợp lệ trước khi gửi
                vec = get_vector(data['img'])
                if not vec or len(vec) == 0:
                    raise ValueError("Không tạo được vector AI cho hình ảnh này")

                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                
                # GỬI DỮ LIỆU
                supabase.table("ai_data").upsert({
                    "id": unique_id, 
                    "file_name": f.name, 
                    "vector": vec,
                    "spec_json": data['all_specs'] if data['all_specs'] else {}, # Đảm bảo luôn là dict
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
                
                logs.append({"File": f.name, "Status": "✅ Success"})
            else:
                logs.append({"File": f.name, "Status": "❌ No Data Extracted"})
        except Exception as e:
            # Ghi lại lỗi chi tiết để kiểm tra
            logs.append({"File": f.name, "Status": f"⚠️ Error: {str(e)}"})
            
    st.session_state['sync_results'] = logs
    st.rerun() # Refresh để cập nhật số SKU ngay lập tức

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Target PDF:", type="pdf")
    if f_audit:
        target = extract_data(f_audit.getvalue(), scan_all=False)
        if target:
            all_db = [r for i in range(0, count, 1000) for r in supabase.table("ai_data").select("id, vector, file_name").range(i, i+999).execute().data]
            df_db = pd.DataFrame(all_db)
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            cols = st.columns(4)
            cols[0].image(target['img'], caption="TARGET")
            for i, (idx, row) in enumerate(top_3.iterrows()):
                det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                if det:
                    with cols[i+1]:
                        st.image(det[0]['image_url'], caption=f"{row['sim']:.1%}")
                        if st.button(f"SELECT {i+1}", key=row['id']): st.session_state['sel_audit'] = {**row.to_dict(), **det[0]}

            if st.session_state['sel_audit']:
                sel = st.session_state['sel_audit']
                st.success(f"Comparing: {sel['file_name']}")
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}"):
                        m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(m_sz[0], {}) if m_sz else {}
                        rows = [{"POM": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        st.table(pd.DataFrame(rows))

elif mode == "🔄 Version Control":
    st.subheader("🔄 Direct Comparison (All Pages Scan)")
    c1, c2 = st.columns(2)
    with c1: f_a = st.file_uploader("File A (Old):", type="pdf", key="fa")
    with c2: f_b = st.file_uploader("File B (New):", type="pdf", key="fb")
    if f_a and f_b:
        if st.button("RUN COMPARISON", use_container_width=True):
            d_a = extract_data(f_a.getvalue(), scan_all=True) # So sánh mới quét tất cả
            d_b = extract_data(f_b.getvalue(), scan_all=True)
            if d_a and d_b:
                report = []
                for sz, specs_b in d_b['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        specs_a = d_a['all_specs'].get(sz, {})
                        rows = []
                        for p, vb in specs_b.items():
                            m_p = get_close_matches(p, list(specs_a.keys()), 1, 0.6)
                            va = specs_a.get(m_p[0], 0) if m_p else 0
                            rows.append({"POM": p, "Old (A)": va, "New (B)": vb, "Diff": vb - va})
                            report.append({"Size": sz, "POM": p, "Old": va, "New": vb, "Diff": vb - va})
                        st.table(pd.DataFrame(rows).style.format({"Diff": "{:+.2f}"}))
                st.download_button("Download Report", to_excel(pd.DataFrame(report)), "compare.xlsx")
