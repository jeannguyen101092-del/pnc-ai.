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
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size; img = img.crop((w*0.05, h*0.05, w*0.95, h*0.7))
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()
    except: return None

def parse_val(t):
    try:
        if not t: return 0
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        # Loại bỏ các ký tự gây nhiễu trong cột thông số
        if any(x in t for x in ["wash", "color", "label", "page", "style"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 # Bỏ qua mã POM B101...
        
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.01 <= val < 200 else 0
        return 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER (QUÉT SẠCH TRANG THÔNG SỐ) =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    # Mẫu nhận diện Size cột: S, M, L, 2, 4, 30, 32...
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|3xl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    
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
                df_w['y_grid'] = df_w['top'].round(0)
                
                # BƯỚC 1: Tìm dòng tiêu đề để xác định tọa độ các cột Size
                size_cols = []
                found_header = False
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    # Chỉ bắt đầu lấy từ trang có chữ Size, Spec hoặc Measurement
                    if any(x in line_txt for x in ["size", "spec", "measurement", "adopted"]):
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and txt not in ["tol", "um", "(+)", "(-)"]:
                                size_cols.append({"sz": txt.upper(), "x_mid": (row['x0'] + row['x1']) / 2})
                        if size_cols: 
                            found_header = True
                            break
                
                if not found_header: continue # Nếu trang này không phải trang thông số thì bỏ qua

                # BƯỚC 2: Quét TẤT CẢ các dòng bên dưới tiêu đề
                for y, group in df_w.groupby('y_grid'):
                    sorted_row = group.sort_values('x0')
                    # Tên điểm đo: tất cả chữ nằm bên trái (thường x1 < 350)
                    pom_parts = sorted_row[sorted_row['x1'] < 380]['text'].tolist()
                    pom_name = " ".join(pom_parts).strip()
                    
                    # Bỏ qua các dòng tiêu đề hoặc dòng trống
                    if not pom_name or any(x in pom_name.lower() for x in ["cover page", "construction", "image", "date", "style"]):
                        continue

                    # Hốt tất cả các cột Size đã định vị
                    found_any_val = False
                    for col in size_cols:
                        # Lấy số nằm đúng trục dọc của cột Size (sai số 20px)
                        cell_data = sorted_row[(sorted_row['x0'] < col['x_mid'] + 20) & (sorted_row['x1'] > col['mid'] - 20) if 'mid' in col else (sorted_row['x0'] < col['x_mid'] + 20) & (sorted_row['x1'] > col['x_mid'] - 20)]
                        val = parse_val(" ".join(cell_data['text']))
                        if val > 0:
                            if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                            all_specs[col['sz']][pom_name] = val
                            found_any_val = True
                            
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Models in Repo", f"{res_count.count or 0} SKUs")
    st.write(f"💾 **Storage:** {(res_count.count or 0)*0.08:.1f}MB / 1024MB")
    st.progress(min((res_count.count or 0)*0.08/1024, 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        for f in new_files:
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": get_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(f"lib_{f_hash}.webp")
                }).execute()
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        if target and target['img']:
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            if res.data:
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                valid_rows = [r for r in res.data if r['vector'] and len(r['vector']) == 512]
                if valid_rows:
                    df = pd.DataFrame(valid_rows)
                    df['sim'] = cosine_similarity(t_vec, np.array(df['vector'].tolist())).flatten()
                    top_3 = df.sort_values('sim', ascending=False).head(3)
                    cols = st.columns(4)
                    cols[0].image(target['img'], caption="TARGET", use_container_width=True)
                    for i, (idx, row) in enumerate(top_3.iterrows()):
                        det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                        if det:
                            with cols[i+1]:
                                st.image(det[0]['image_url'], caption=f"Match: {row['sim']:.1%}")
                                if st.button(f"CHỌN {i+1}", key=f"s_{idx}"):
                                    st.session_state['sel_audit'] = {**row.to_dict(), **det[0]}

            sel = st.session_state['sel_audit']
            if sel:
                st.divider(); st.success(f"📈 So sánh với: **{sel['file_name']}**")
                audit_dfs, sheet_names = [], []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(m_sz[0] if m_sz else "", {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows: r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                        df_sz = pd.DataFrame(rows); st.table(df_sz)
                        audit_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất Excel Audit", to_excel(audit_dfs, sheet_names), "Audit_Report.xlsx")

elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh 2 file PDF (Quét Toàn Bộ Dòng)")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key="v1")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key="v2")
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh toàn diện", use_container_width=True):
            d1, d2 = extract_full_data(f1.getvalue()), extract_full_data(f2.getvalue())
            if d1 and d2:
                st.session_state['ver_results'] = {"d1": d1, "d2": d2, "f1_name": f1.name, "f2_name": f2.name}

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider(); col_a, col_b = st.columns(2)
        col_a.image(vr['d1']['img'], caption=f"A: {vr['f1_name']}", use_container_width=True)
        col_b.image(vr['d2']['img'], caption=f"B: {vr['f2_name']}", use_container_width=True)
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())), key=lambda x: str(x))
        version_dfs, ver_sheets = [], []
        for sz in all_sz:
            with st.expander(f"SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = [{"Point": p, "Ver A": s1.get(p,0), "Ver B": s2.get(p,0), "Diff": f"{s2.get(p,0)-s1.get(p,0):+.3f}", "Status": "✅" if s1.get(p,0)==s2.get(p,0) else "⚠️"} for p in poms]
                df_sz = pd.DataFrame(rows); st.table(df_sz); version_dfs.append(df_sz); ver_sheets.append(sz)
        st.download_button("📥 Xuất Excel So Sánh", to_excel(version_dfs, ver_sheets), "Comparison.xlsx")
