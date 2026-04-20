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
    st.subheader("🔄 So sánh Toàn diện (Quét mọi trang PDF)")

    # --- NÚT XÓA & LÀM MỚI ---
    if st.button("🗑️ Xoá file & Làm mới", use_container_width=True):
        st.session_state['up_key'] += 1
        st.session_state['ver_results'] = None
        st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key=f"v2_{st.session_state['up_key']}")

    # --- HÀM QUÉT SIÊU VIỆT ---
    def scan_all_pages(content):
        all_data = {}
        first_img = None
        try:
            # Lấy ảnh trang 1 làm đại diện
            doc = fitz.open(stream=content, filetype="pdf")
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            first_img = pix.tobytes("png")
            doc.close()

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df = pd.DataFrame(words)
                    df['y'] = df['top'].round(0)

                    # 1. Tìm hàng Size tiêu đề
                    sz_cols = []
                    for y, gp in df.groupby('y'):
                        txt_line = " ".join(gp.sort_values('x0')['text']).lower()
                        if any(x in txt_line for x in ["size", "spec", "adopted", "tol"]):
                            for _, r in gp.iterrows():
                                t = r['text'].strip().upper()
                                # Nhận diện Size: số hoặc chữ XS-XXL, bỏ qua rác
                                if re.match(r'^(XS|S|M|L|XL|XXL|\d{1,3})$', t):
                                    sz_cols.append({"sz": t, "x0": r['x0']-12, "x1": r['x1']+12})
                            if len(sz_cols) >= 2: break # Tìm thấy ít nhất 2 size mới coi là bảng

                    if not sz_cols: continue

                    # 2. Bóc POM Name & Số liệu
                    for y, gp in df.groupby('y'):
                        s_gp = gp.sort_values('x0')
                        # Tên POM nằm bên trái cột size đầu tiên
                        pom = " ".join(s_gp[s_gp['x1'] < sz_cols[0]['x0']]['text']).strip()
                        
                        if 3 < len(pom) < 80 and not any(x in pom.upper() for x in ["DATE", "PAGE", "STYLE", "FABRIC", "IMAGE"]):
                            for col in sz_cols:
                                cell = s_gp[(s_gp['x0'] >= col['x0']) & (s_gp['x1'] <= col['x1'])]
                                if not cell.empty:
                                    raw_num = " ".join(cell['text'])
                                    # Xử lý số, phân số (1/2, 1/4...)
                                    try:
                                        if "/" in raw_num:
                                            nums = re.findall(r"(\d+)?\s?(\d+)/(\d+)", raw_num)
                                            val = sum([float(n[0] or 0) + int(n[1])/int(n[2]) for n in nums])
                                        else:
                                            val = float(re.findall(r"\d+\.?\d*", raw_num)[0])
                                        
                                        if 0.1 <= val < 200:
                                            if col['sz'] not in all_data: all_data[col['sz']] = {}
                                            all_data[col['sz']][pom] = val
                                    except: continue
            return {"specs": all_data, "img": first_img}
        except: return None

    # --- CHẠY SO SÁNH ---
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh TOÀN BỘ TRANG", use_container_width=True):
            with st.spinner("Đang tìm bảng POM ở tất cả các trang..."):
                d1 = scan_all_pages(f1.getvalue())
                d2 = scan_all_pages(f2.getvalue())
                if d1 and d2 and (d1['specs'] or d2['specs']):
                    st.session_state['ver_results'] = {"d1": d1, "d2": d2}
                else:
                    st.error("❌ Không tìm thấy bảng thông số nào. Hãy kiểm tra lại file PDF.")

    # --- HIỂN THỊ ---
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider()
        
        all_sz = sorted(list(set(vr['d1']['specs'].keys()) | set(vr['d2']['specs'].keys())))
        
        def color_diff(val):
            if val == "❌ Lệch": return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
            if val == "✅ Khớp": return 'background-color: #ccffcc; color: #006600;'
            return 'background-color: #fff3cd; color: #856404;'

        for sz in all_sz:
            with st.expander(f"📊 SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['specs'].get(sz, {}), vr['d2']['specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []
                for p in poms:
                    v1, v2 = s1.get(p), s2.get(p)
                    if v1 is not None and v2 is not None:
                        diff = round(v2 - v1, 3)
                        res = "✅ Khớp" if abs(diff) < 0.001 else "❌ Lệch"
                        df_txt = f"{diff:+.3f}"
                    else: df_txt, res = "N/A", "⚠️ Thiếu"
                    rows.append({"POM Name": p, "Bản A": v1, "Bản B": v2, "Lệch": df_txt, "Kết quả": res})
                
                df_sz = pd.DataFrame(rows)
                try: st.dataframe(df_sz.style.map(color_diff, subset=['Kết quả']), use_container_width=True)
                except: st.dataframe(df_sz.style.applymap(color_diff, subset=['Kết quả']), use_container_width=True)

        if st.download_button("📥 Xuất Excel So Sánh", to_excel([], []), "Full_Compare.xlsx", use_container_width=True):
            st.success("Đã xuất xong!")
