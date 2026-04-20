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
    st.subheader("🔄 So sánh Toàn diện (Đã sửa lỗi không tìm thấy bảng)")

    # --- HÀM XỬ LÝ SỐ & PHÂN SỐ ---
    def parse_any_num(text):
        try:
            text = text.strip().lower()
            mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', text)
            if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
            num = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if num: return float(num[0])
            return None
        except: return None

    # --- THUẬT TOÁN DÒ TÌM BẢNG KHÔNG CẦN CHỮ "SIZE" ---
    def auto_detect_express_table(content):
        all_specs = {}
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df = pd.DataFrame(words)
                    df['y'] = (df['top'] / 2.5).round(0) * 2.5
                    
                    # 1. DÒ TÌM DÒNG HEADER (Dòng chứa các Size liên tiếp)
                    size_lanes = []
                    for y, gp in df.groupby('y'):
                        sorted_gp = gp.sort_values('x0')
                        candidates = []
                        for _, r in sorted_gp.iterrows():
                            t = r['text'].strip().upper()
                            # Kiểm tra nếu là Size số (28-40) hoặc Size chữ (XS-XXXL)
                            if (re.match(r'^\d{2,3}$', t) or t in ["XS","S","M","L","XL","XXL","XXXL"]) and r['x0'] > 200:
                                candidates.append({"sz": t, "x0": r['x0']-12, "x1": r['x1']+12})
                        
                        # Nếu một dòng có từ 3 Size trở lên nằm sát nhau -> Đây chính là Header
                        if len(candidates) >= 3:
                            size_lanes = candidates
                            break

                    if not size_lanes: continue # Trang này không có bảng -> Tìm trang tiếp theo

                    # 2. QUÉT DỮ LIỆU DƯỚI HEADER
                    for y, gp in df.groupby('y'):
                        sorted_gp = gp.sort_values('x0')
                        # POM Description nằm bên trái (x1 < cột Size đầu tiên)
                        first_sz_x = size_lanes[0]['x0']
                        pom_parts = sorted_gp[sorted_gp['x1'] < first_sz_x]['text'].values
                        pom_name = " ".join(pom_parts).strip()
                        
                        if 3 < len(pom_name) < 100 and not any(x in pom_name.upper() for x in ["DATE", "PAGE", "STYLE"]):
                            for lane in size_lanes:
                                cell = sorted_gp[(sorted_gp['x0'] >= lane['x0']) & (sorted_gp['x1'] <= lane['x1'])]
                                if not cell.empty:
                                    v = parse_any_num(" ".join(cell['text']))
                                    if v is not None and 0.1 <= v < 300:
                                        if lane['sz'] not in all_specs: all_specs[lane['sz']] = {}
                                        all_specs[lane['sz']][pom_name] = v
            return all_specs
        except: return None

    # --- UI & TABS ---
    if st.button("🗑️ Làm mới hệ thống"):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A)", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B)", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ BẮT ĐẦU SO SÁNH", use_container_width=True):
            with st.spinner("Đang tự động dò tìm bảng thông số..."):
                s1 = auto_detect_express_table(f1.getvalue())
                s2 = auto_detect_express_table(f2.getvalue())
                if s1 and s2:
                    st.session_state['ver_results'] = {"s1": s1, "s2": s2}
                else:
                    st.error("❌ Vẫn không tìm thấy bảng Specs. Hãy đảm bảo PDF không phải là dạng ảnh quét (Scan).")

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        s1, s2 = vr['s1'], vr['s2']
        
        # Sắp xếp Size theo thứ tự may mặc
        sz_order = ["28","29","30","31","32","33","34","36","38","40","XS","S","M","L","XL","XXL"]
        all_sz = [s for s in sz_order if s in s1 or s in s2] or sorted(list(set(s1.keys()) | set(s2.keys())))

        tabs = st.tabs([f"Size {sz}" for sz in all_sz])
        for i, sz in enumerate(all_sz):
            with tabs[i]:
                d1, d2 = s1.get(sz, {}), s2.get(sz, {})
                poms = sorted(list(set(d1.keys()) | set(d2.keys())))
                rows = []
                for p in poms:
                    v1, v2 = d1.get(p), d2.get(p)
                    diff = round(v2 - v1, 3) if (v1 is not None and v2 is not None) else None
                    rows.append({"POM": p, "Bản A": v1, "Bản B": v2, "Lệch": f"{diff:+.3f}" if diff is not None else "N/A", "Kết quả": "✅ Khớp" if (diff is not None and abs(diff) < 0.001) else "❌ Lệch"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=600)
