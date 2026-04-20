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
    st.subheader("🔄 So sánh Toàn diện (Đã Fix lỗi bốc nhầm số & Ẩn số Bản A)")

    # --- HÀM BÓC TÁCH DỮ LIỆU SIÊU SẠCH ---
    def get_clean_specs(content):
        data_list = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df = pd.DataFrame(words)
                    df['y'] = (df['top'] / 2).round(0) * 2
                    
                    # 1. Dò tìm Header Size (Chỉ lấy cột Size thực sự)
                    size_lanes = []
                    for y, gp in df.groupby('y'):
                        sorted_gp = gp.sort_values('x0')
                        candidates = []
                        for _, r in sorted_gp.iterrows():
                            t = r['text'].strip().upper()
                            # Chỉ lấy XS-XXXL hoặc số nguyên nhỏ 0-42
                            if (re.match(r'^(XXS|XS|S|M|L|XL|XXL|1X|2X|3X|[0-9]{1,2}|000|00|0)$', t)) and r['x0'] > 220:
                                if t not in ["TOL", "GRADE", "SPEC", "POM"]:
                                    candidates.append({"sz": t, "x0": r['x0']-10, "x1": r['x1']+10})
                        if len(candidates) >= 2:
                            size_lanes = candidates
                            break 

                    if not size_lanes: continue

                    # 2. Bóc tách dữ liệu
                    first_sz_x = min([c['x0'] for c in size_lanes])
                    for y, gp in df.groupby('y'):
                        sorted_gp = gp.sort_values('x0')
                        # Lấy POM ở vùng bên trái (x < 220)
                        pom_raw = " ".join(sorted_gp[sorted_gp['x1'] < first_sz_x]['text']).strip()
                        
                        # KIỂM TRA POM: Phải có ít nhất 2 chữ cái, không được chỉ toàn số
                        if len(re.sub(r'[^a-zA-Z]', '', pom_raw)) > 2:
                            for col in size_lanes:
                                cell = sorted_gp[(sorted_gp['x0'] >= col['x0']) & (sorted_gp['x1'] <= col['x1'])]
                                if not cell.empty:
                                    txt_val = " ".join(cell['text'])
                                    # Parse số (hỗ trợ phân số 1/2, 1/4)
                                    m = re.findall(r"(\d+)\s+(\d+)/(\d+)|(\d+)/(\d+)|(\d+\.?\d*)", txt_val)
                                    val = None
                                    if m:
                                        tup = m[0]
                                        if tup[0]: val = float(tup[0]) + int(tup[1])/int(tup[2])
                                        elif tup[3]: val = int(tup[3])/int(tup[4])
                                        elif tup[5]: val = float(tup[5])
                                    
                                    if val is not None:
                                        data_list.append({"Size": col['sz'], "POM": pom_raw, "Value": val})
            return pd.DataFrame(data_list)
        except: return pd.DataFrame()

    # --- UI & SO SÁNH ---
    if st.button("🗑️ Làm mới hệ thống"):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản A (Cũ)", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản B (Mới)", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ CHẠY SO SÁNH CHUẨN X-Y", use_container_width=True):
            with st.spinner("Đang bóc tách dữ liệu..."):
                df_a = get_clean_specs(f1.getvalue())
                df_b = get_clean_specs(f2.getvalue())
                if not df_a.empty and not df_b.empty:
                    st.session_state['ver_results'] = {"a": df_a, "b": df_b}
                else: st.error("❌ Không tìm thấy bảng thông số hợp lệ.")

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        df_a, df_b = vr['a'], vr['b']
        
        # Lấy danh sách Size chuẩn (bỏ qua số trang tào lao)
        all_sizes = sorted(list(set(df_a['Size'].unique()) | set(df_b['Size'].unique())))
        tabs = st.tabs([f"Size {s}" for s in all_sizes])
        
        for i, sz in enumerate(all_sizes):
            with tabs[i]:
                # Lọc và so khớp 2 bản
                s_a = df_a[df_a['Size'] == sz][['POM', 'Value']].rename(columns={'Value': 'Bản A'})
                s_b = df_b[df_b['Size'] == sz][['POM', 'Value']].rename(columns={'Value': 'Bản B'})
                
                # Merge chính xác theo tên POM
                res = pd.merge(s_a, s_b, on='POM', how='outer').fillna(np.nan)
                
                # Tính toán
                def calc_diff(row):
                    if pd.isna(row['Bản A']) or pd.isna(row['Bản B']): return "N/A", "⚠️ Thiếu"
                    diff = round(row['Bản B'] - row['Bản A'], 3)
                    status = "✅ Khớp" if abs(diff) < 0.01 else "❌ Lệch"
                    return f"{diff:+.3f}", status
                
                res[['Lệch', 'Kết quả']] = res.apply(lambda r: pd.Series(calc_diff(r)), axis=1)
                
                # Hiển thị
                res = res.sort_values('POM')
                st.dataframe(res, use_container_width=True, height=500)
