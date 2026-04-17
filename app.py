import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
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
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(base.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size; img = img.crop((w*0.05, h*0.05, w*0.95, h*0.7))
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()
    except: return None

# --- HÀM XỬ LÝ SỐ ĐO (ĐÃ FIX LỖI QUÉT NHẦM MÃ POM B101, B200) ---
def parse_val(t):
    try:
        # Làm sạch chuỗi
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page"]): return 0
        
        # FIX: Nếu chuỗi là mã điểm đo (bắt đầu bằng chữ + số như B101, D200) -> Bỏ qua không lấy số này
        if re.match(r'^[a-z]\d+', t): return 0
        
        # 1. Xử lý hỗn số kiểu: 31 1/2
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        
        # 2. Xử lý phân số kiểu: 1/2, 3/4
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        
        # 3. Xử lý số thập phân hoặc số nguyên
        num_match = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num_match:
            val = float(num_match[0])
            # Chỉ lấy giá trị thực tế của thông số may mặc (thường < 150)
            return val if val < 150 else 0
        return 0
    except: return 0

def detect_category(filename):
    fn = filename.upper()
    if any(k in fn for k in ["SHORT", "SKIRT"]): return "SHORT"
    if any(k in fn for k in ["PANT", "JEAN", "TROUSER", "LEG"]): return "PANT"
    return "TOP/OTHER"

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names): df.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()

# ================= 3. SCRAPER (ĐÃ FIX ĐỂ TÁCH BIỆT MÃ VÀ SỐ ĐO) =================
def extract_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_pil.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue(); doc.close()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = (df_w['top'] / 2).round() * 2
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text'])
                    # Chỉ lấy các giá trị số đo hợp lệ từ phần sau của dòng
                    line_vals = [parse_val(w) for w in line_txt.split() if parse_val(w) > 0]
                    
                    if any(kw in line_txt.upper() for kw in POM_KWS) and line_vals:
                        # Tên điểm đo: Loại bỏ các con số ở cuối dòng
                        pom_name = re.sub(r'[\d./\s]+$', '', line_txt).strip()
                        if len(pom_name) > 2:
                            for i, val in enumerate(line_vals):
                                s_key = f"Size_{i+1}"
                                if s_key not in all_specs: all_specs[s_key] = {}
                                all_specs[s_key][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("PPJ GROUP")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Repository", f"{res_count.count or 0} SKUs")
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE"):
        for f in new_files:
            data = extract_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({"id":str(uuid.UUID(f_hash)), "file_name":f.name, "vector":get_vector(data['img']), "spec_json":data['all_specs'], "image_url":supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_data(f_audit.getvalue())
        if target and target['img']:
            target_cat = detect_category(f_audit.name)
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            if res.data:
                t_vec = get_vector(target['img'])
                filtered = [r for r in res.data if detect_category(r['file_name']) == target_cat]
                df = pd.DataFrame(filtered if filtered else res.data)
                df = df[df['vector'].apply(lambda v: len(v) == len(t_vec))]
                
                if not df.empty:
                    df['sim'] = cosine_similarity(np.array(t_vec).reshape(1,-1), np.array(df['vector'].tolist())).flatten()
                    top_3 = df.sort_values('sim', ascending=False).head(3)
                    st.subheader(f"🎯 Kết quả tìm kiếm (Nhóm: {target_cat})")
                    cols = st.columns(4)
                    cols.image(target['img'], caption="TARGET PDF", use_container_width=True)
                    for i, (idx, row) in enumerate(top_3.iterrows()):
                        with cols[i+1]:
                            det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                            st.image(det['image_url'], caption=f"Match: {row['sim']:.1%}")
                            if st.button(f"CHỌN {i+1}", key=f"btn_{row['id']}"): st.session_state['sel_audit'] = {**row.to_dict(), **det}

            sel = st.session_state['sel_audit']
            if sel:
                st.divider(); st.success(f"📈 So sánh với: **{sel['file_name']}**")
                all_dfs, sheet_names = [], []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(m_sz, {}) if m_sz else {}
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p, 0) if m_p else 0
                            rows.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{v-rv:+.3f}"})
                        df_sz = pd.DataFrame(rows); st.table(df_sz)
                        all_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất báo cáo Excel", to_excel(all_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh 2 file PDF mới")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key="v1")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key="v2")
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh", use_container_width=True):
            d1, d2 = extract_data(f1.getvalue()), extract_data(f2.getvalue())
            if d1 and d2:
                st.divider(); c_a, c_b = st.columns(2)
                c_a.image(d1['img'], caption="Bản A", use_container_width=True); c_b.image(d2['img'], caption="Bản B", use_container_width=True)
                all_sz = sorted(list(set(d1['all_specs'].keys()) | set(d2['all_specs'].keys())))
                all_dfs, sheet_names = [], []
                for sz in all_sz:
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        s1, s2 = d1['all_specs'].get(sz, {}), d2['all_specs'].get(sz, {})
                        comp = [{"Point": p, "Old": s1.get(p,0), "New": s2.get(p,0), "Diff": f"{s2.get(p,0)-s1.get(p,0):+.3f}", "Status": "✅" if s1.get(p,0)==s2.get(p,0) else "⚠️"} for p in sorted(list(set(s1.keys()) | set(s2.keys())))]
                        df_sz = pd.DataFrame(comp); st.table(df_sz); all_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất Excel", to_excel(all_dfs, sheet_names), "Comparison.xlsx")
