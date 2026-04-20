import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, requests
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'up_key' not in st.session_state: st.session_state['up_key'] = 0
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None

# ================= 2. AI CORE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        img = img.crop((w*0.20, h*0.12, w*0.80, h*0.50)) 
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.5).convert('RGB')
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

# ================= 3. SCRAPER =================
def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        trash = ["poly", "bag", "twill", "label", "button", "thread", "frisbee", "seaman", "paper"]
        if not t or any(x in t for x in trash): return 0
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.2 <= val < 150 else 0
        return 0
    except: return 0

def extract_full_data(file_content):
    all_specs, img_bytes = {}, None
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    if any(x in line_txt for x in ["size", "spec", "adopted"]):
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and row['x0'] > 200:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0']-10, "x1": row['x1']+10})
                        if size_cols: break
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    pom_name = " ".join(sorted_group[sorted_group['x1'] < 300]['text']).strip()
                    if 3 < len(pom_name) < 65 and not any(x in pom_name.upper() for x in ["STYLE", "DATE", "FABRIC", "PAGE"]):
                        for col in size_cols:
                            cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 4. SIDEBAR (INCLUDES STORAGE METRIC) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        res_db = supabase.table("ai_data").select("id", count="exact").execute()
        count = res_db.count or 0
    except: count = 0
    
    # Hiển thị SKUs và Dung lượng
    st.metric("Models in Repo", f"{count} SKUs")
    storage_mb = count * 0.08  # Ước tính 80KB mỗi SKU
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()
    
    with st.expander("⚙️ Bảo trì & Nâng cấp AI"):
        if st.button("🚀 BẮT ĐẦU", use_container_width=True):
            items = (supabase.table("ai_data").select("id, image_url").execute()).data
            if items:
                p_bar = st.progress(0)
                for i, item in enumerate(items):
                    try:
                        r = requests.get(item['image_url'], timeout=10)
                        if r.status_code == 200:
                            v = get_vector(r.content)
                            if v: supabase.table("ai_data").update({"vector": v}).eq("id", item['id']).execute()
                        p_bar.progress((i+1)/len(items))
                    except: continue
                st.success("Xong!"); st.rerun()

    st.divider()
    st.subheader("📥 Nạp kho mẫu mới")
    up_new = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"side_up_{st.session_state['up_key']}")
    if up_new and st.button("🚀 ĐẨY VÀO KHO", use_container_width=True):
        p_bar = st.progress(0)
        for i, f in enumerate(up_new):
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                fname = f"{uuid.uuid4()[:8]}_{f.name.replace(' ', '_')}.webp"
                supabase.storage.from_(BUCKET).upload(f"sketches/{fname}", data['img'])
                img_url = supabase.storage.from_(BUCKET).get_public_url(f"sketches/{fname}")
                supabase.table("ai_data").insert({
                    "file_name": f.name, "image_url": img_url, 
                    "vector": get_vector(data['img']), "specs": data['all_specs']
                }).execute()
            p_bar.progress((i+1)/len(up_new))
        st.success("Đã nạp xong!"); st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["Audit Mode", "Version Control"], horizontal=True)

if mode == "Audit Mode":
    st.subheader("🔍 Tìm kiếm mẫu tương đồng")
    target_file = st.file_uploader("Upload Target PDF", type=['pdf'], key=f"aud_{st.session_state['up_key']}")
    if target_file:
        with st.spinner("Đang tìm kiếm..."):
            t_data = extract_full_data(target_file.getvalue())
            if t_data and t_data['img']:
                t_vec = get_vector(t_data['img'])
                res = supabase.table("ai_data").select("file_name, image_url, vector").execute()
                db_items = res.data
                if db_items and t_vec:
                    scores = []
                    for item in db_items:
                        if item['vector']:
                            sim = cosine_similarity([t_vec], [item['vector']])
                            scores.append({"name": item['file_name'], "url": item['image_url'], "score": sim[0][0]})
                    top_8 = sorted(scores, key=lambda x: x['score'], reverse=True)[:8]
                    st.divider()
                    cols = st.columns(4)
                    for idx, item in enumerate(top_8):
                        with cols[idx % 4]:
                            st.image(item['url'], use_container_width=True)
                            st.caption(f"**{item['name']}**")
                            st.info(f"Độ giống: {item['score']:.1%}")

elif mode == "Version Control":
    st.subheader("🔄 So sánh Toàn diện (Mọi trang + POM Name + Description)")

    # --- HÀM QUÉT ĐA TRANG THÔNG MINH ---
    def enhanced_multi_page_extract(content):
        specs, img_list = {}, []
        doc = fitz.open(stream=content, filetype="pdf")
        # Lấy ảnh preview của trang 1 và trang 2 (nếu có) để xem mẫu
        for i in range(min(len(doc), 2)):
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_list.append(pix.tobytes("png"))
        doc.close()

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                h_p = page.height
                words = page.extract_words()
                if not words: continue
                df = pd.DataFrame(words)
                df['y'] = df['top'].round(0)
                
                # 1. Tìm dòng Header chứa Size hoặc POM/Description
                sz_cols = []
                for y, gp in df.groupby('y'):
                    txt = " ".join(gp.sort_values('x0')['text']).upper()
                    # Nhận diện dòng tiêu đề: chứa POM, Description hoặc Size
                    if any(x in txt for x in ["POM", "DESCRIPTION", "SIZE", "SPEC", "ADOPTED"]):
                        for _, r in gp.iterrows():
                            t_clean = r['text'].strip().upper()
                            # Lưu vị trí các cột Size (số hoặc chữ XS-XXL)
                            if re.match(r'^(XS|S|M|L|XL|XXL|\d+|[A-Z]?\d+-\d+)$', t_clean) and r['x0'] > 150:
                                sz_cols.append({"sz": t_clean, "x0": r['x0']-12, "x1": r['x1']+12})
                        if sz_cols: continue 

                # 2. Bóc tách dữ liệu POM/Description
                if not sz_cols: continue # Trang này không có bảng thông số -> bỏ qua

                for y, gp in df.groupby('y'):
                    s_gp = gp.sort_values('x0')
                    # POM Name/Description thường nằm bên trái (x1 < cột size đầu tiên)
                    left_limit = min([c['x0'] for c in sz_cols])
                    pom_parts = s_gp[s_gp['x1'] <= left_limit]['text'].values
                    pom_desc = " ".join(pom_parts).strip()
                    
                    # Lọc bỏ rác: bỏ dòng quá ngắn hoặc dòng tiêu đề lặp lại
                    if 3 < len(pom_desc) < 100 and not any(x in pom_desc.upper() for x in ["SIZE", "DATE", "PAGE", "STYLE", "FABRIC"]):
                        for c in sz_cols:
                            cell = s_gp[(s_gp['x0'] >= c['x0']) & (s_gp['x1'] <= c['x1'])]
                            if not cell.empty:
                                # Parse giá trị số (hỗ trợ cả phân số 1/2, 1/4...)
                                raw_val = " ".join(cell['text'])
                                val = 0
                                try:
                                    if "/" in raw_val:
                                        pts = raw_val.split()
                                        val = sum([float(p.split('/')[0])/float(p.split('/')[1]) if '/' in p else float(p) for p in pts])
                                    else: val = float(re.findall(r"[-+]?\d*\.\d+|\d+", raw_val)[0])
                                except: continue
                                
                                if 0.1 <= val < 200: # Chỉ lấy số đo may mặc hợp lệ
                                    if c['sz'] not in specs: specs[c['sz']] = {}
                                    specs[c['sz']][pom_desc] = val
        return {"all_specs": specs, "imgs": img_list}

    # --- UI PHẦN SO SÁNH ---
    if st.button("🗑️ Xoá file & Làm mới", use_container_width=True):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ Bắt đầu quét TOÀN BỘ TRANG", use_container_width=True):
            with st.spinner("Đang lục tìm POM Description ở tất cả các trang..."):
                d1 = enhanced_multi_page_extract(f1.getvalue())
                d2 = enhanced_multi_page_extract(f2.getvalue())
                if d1['all_specs'] and d2['all_specs']:
                    st.session_state['ver_results'] = {"d1": d1, "d2": d2}
                else: st.error("❌ Không tìm thấy bảng thông số (POM/Description) ở bất kỳ trang nào.")

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider()

        # Lấy tất cả Size tìm được từ mọi trang
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        version_dfs, ver_sheets = [], []

        def color_st(val):
            if val == "❌ Lệch": return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
            if val == "✅ Khớp": return 'background-color: #ccffcc; color: #006600;'
            return 'background-color: #fff3cd; color: #856404;'

        for sz in all_sz:
            with st.expander(f"📊 BẢNG THÔNG SỐ SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                # Gom tất cả POM Name / Description
                all_poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []
                for p in all_poms:
                    v1, v2 = s1.get(p), s2.get(p)
                    if v1 is not None and v2 is not None:
                        diff = round(v2 - v1, 3)
                        st_v = "✅ Khớp" if abs(diff) < 0.001 else "❌ Lệch"
                        dt = f"{diff:+.3f}"
                    else: dt, st_v = "N/A", "⚠️ Thiếu"
                    rows.append({"POM / Description": p, "Bản A": v1, "Bản B": v2, "Lệch": dt, "Kết quả": st_v})
                
                df = pd.DataFrame(rows)
                try: st.dataframe(df.style.map(color_st, subset=['Kết quả']), use_container_width=True)
                except: st.dataframe(df.style.applymap(color_st, subset=['Kết quả']), use_container_width=True)
                version_dfs.append(df); ver_sheets.append(f"Size_{sz}")

        if version_dfs:
            st.download_button("📥 Xuất Excel So Sánh Toàn Bộ", to_excel(version_dfs, ver_sheets), "Full_Comparison.xlsx", use_container_width=True)
