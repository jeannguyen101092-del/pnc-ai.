import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, json
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Pro Auditor", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (SIẾU NHẬN DIỆN & CHỐNG LỖI) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Cắt ảnh cực sâu: Bỏ 25% đầu trang (Bảng biểu) và lấy vùng trung tâm
        w, h = img.size
        img = img.crop((w*0.1, h*0.25, w*0.9, h*0.75)) 
        # Tăng tương phản tối đa để AI thấy rõ đường nét áo/quần
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(3.0).convert('RGB')

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()
    except: return None

def get_category_tags(name):
    n = str(name).upper()
    is_top = any(k in n for k in ["TOP", "SHIRT", "JACKET", "TEE", "POLO", "VEST", "COAT", "HOODIE"])
    is_btm = any(k in n for k in ["PANT", "SHORT", "JEAN", "LEG", "TROUSER", "SKIRT", "BERMUDA"])
    if is_top: return "TOP"
    if is_btm: return "BOTTOM"
    return "UNKNOWN"

def calculate_similarity_engine(target_data, repo_item):
    """Hàm so khớp 3 lớp với bộ lọc lỗi TYPEERROR cực mạnh"""
    t_vec = target_data.get('vector')
    r_vec = repo_item.get('vector')
    
    # KIỂM TRA AN TOÀN TUYỆT ĐỐI: Bỏ qua nếu dữ liệu không phải là list số
    if not isinstance(t_vec, list) or not isinstance(r_vec, list) or len(t_vec) != len(r_vec):
        return 0.0

    try:
        # Tính độ tương đồng hình ảnh
        sim_visual = float(cosine_similarity(np.array(t_vec).reshape(1,-1), np.array(r_vec).reshape(1,-1)))
        
        # Logic lọc cứng loại sản phẩm (Hard Filter)
        t_cat = get_category_tags(target_data.get('name', ""))
        r_cat = get_category_tags(repo_item.get('file_name', ""))
        
        # Nếu đã xác định rõ loại mà khác nhau thì loại bỏ ngay
        if t_cat != "UNKNOWN" and r_cat != "UNKNOWN" and t_cat != r_cat:
            return 0.0
            
        # Thưởng điểm nếu cùng loại cụ thể
        if t_cat == r_cat and t_cat != "UNKNOWN": sim_visual += 0.2
        
        return min(1.0, sim_visual)
    except:
        return 0.0

# ================= 3. SCRAPER (BẢN CHUẨN) =================
def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        return float(num[0]) if num else 0
    except: return 0

def extract_full_data(file_content, filename=""):
    if not file_content: return None
    all_specs = {}
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_pil.save(buf, format="WEBP", quality=70); img_data = buf.getvalue(); doc.close()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words); df_w['y_grid'] = df_w['top'].round(0)
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    if "size" in line_txt or "adopted" in line_txt:
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and txt not in ["tol", "um", "(+)", "(-)"]:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0']-5, "x1": row['x1']+5})
                        if size_cols: break
                for y, group in df_w.groupby('y_grid'):
                    sorted_row = group.sort_values('x0')
                    line_txt = " ".join(sorted_row['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "CONSTRUCTION"]): continue
                    pom_name = re.sub(r'[\d./\s]+$', '', " ".join(sorted_row[sorted_row['x1'] < 350]['text'])).strip()
                    if len(pom_name) > 3:
                        for col in size_cols:
                            cell = sorted_row[(sorted_row['x0'] >= col['x0']) & (sorted_row['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_data, "vector": get_vector(img_data), "name": filename}
    except: return None

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names): df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Models in Repo", f"{count} SKUs")
    storage_mb = count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        prog = st.progress(0); p_text = st.empty()
        for i, f in enumerate(new_files):
            data = extract_full_data(f.getvalue(), f.name)
            if data:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "spec_json": data['all_specs'], 
                    "vector": data['vector'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            prog.progress((i+1)/len(new_files)); p_text.write(f"Done: {i+1}/{len(new_files)}")
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue(), f_audit.name)
        if target:
            # LẤY DỮ LIỆU VÀ SO KHỚP AN TOÀN
            res = supabase.table("ai_data").select("*").execute()
            if res.data:
                valid_matches = []
                for r in res.data:
                    sim = calculate_similarity_engine(target, r)
                    if sim > 0: # Chỉ hiển thị những mẫu cùng loại
                        valid_matches.append({**r, "sim_score": sim})
                
                df_db = pd.DataFrame(valid_matches).sort_values('sim_score', ascending=False).head(3)
                
                st.subheader(f"🎯 AI Matches (Detected: {get_category_tags(f_audit.name)})")
                cols = st.columns(4)
                cols.image(target['img'], caption="TARGET PDF", use_container_width=True)
                for i, (idx, row) in enumerate(df_db.iterrows()):
                    with cols[i+1]:
                        st.image(row['image_url'], caption=f"Match: {row['sim_score']:.1%}")
                        st.write(f"**{row['file_name']}**")
                        if st.button(f"CHỌN {i+1}", key=f"s_{idx}", use_container_width=True):
                            st.session_state['sel_audit'] = row

            if st.session_state['sel_audit']:
                sel = st.session_state['sel_audit']
                st.divider(); st.success(f"📈 Comparing with: **{sel['file_name']}**")
                audit_dfs, sheet_names = [], []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = sel['spec_json'].get(get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)[0] if get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4) else "", {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows: r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                        df_sz = pd.DataFrame(rows); st.table(df_sz); audit_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất Excel Audit", to_excel(audit_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

elif mode == "🔄 Version Control":
    # Phần Version Control chuẩn của bạn
    st.subheader("🔄 So sánh 2 file PDF mới")
    c1, c2 = st.columns(2)
    f1, f2 = c1.file_uploader("Bản cũ (A):", type="pdf", key="v1"), c2.file_uploader("Bản mới (B):", type="pdf", key="v2")
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh", use_container_width=True):
            d1, d2 = extract_full_data(f1.getvalue(), f1.name), extract_full_data(f2.getvalue(), f2.name)
            if d1 and d2: st.session_state['ver_results'] = {"d1": d1, "d2": d2, "f1_name": f1.name, "f2_name": f2.name}
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider(); col_a, col_b = st.columns(2)
        col_a.image(vr['d1']['img'], caption=f"Bản A", use_container_width=True); col_b.image(vr['d2']['img'], caption=f"Bản B", use_container_width=True)
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())), key=lambda x: str(x))
        version_dfs, ver_sheets = [], []
        for sz in all_sz:
            with st.expander(f"SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = [{"Point": p, "Ver A": s1.get(p,0), "Ver B": s2.get(p,0), "Diff": f"{s2.get(p,0)-s1.get(p,0):+.3f}", "Status": "✅" if s1.get(p,0)==s2.get(p,0) else "⚠️"} for p in sorted(list(set(s1.keys()) | set(s2.keys())))]
                df_sz = pd.DataFrame(rows); st.table(df_sz); version_dfs.append(df_sz); ver_sheets.append(sz)
        st.download_button("📥 Xuất Excel So Sánh", to_excel(version_dfs, ver_sheets), "Comparison.xlsx")
