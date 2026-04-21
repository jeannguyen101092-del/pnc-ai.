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
# ================= 4. SIDEBAR (BẢN FIX LỖI COLUMN SPECS) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    
    try:
        res_db = supabase.table("ai_data").select("id", count="exact").execute()
        current_count = res_db.count or 0
    except:
        current_count = 0
    
    st.metric("Models in Repo", f"{current_count} SKUs")
    storage_mb = current_count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    
    if st.button("🔄 Làm mới số lượng"): st.rerun()

    st.divider()
    st.subheader("📥 Nạp kho mẫu mới")
    up_new = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"side_up_{st.session_state['up_key']}")
    
    if up_new and st.button("🚀 ĐẨY VÀO KHO", use_container_width=True):
        p_bar = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(up_new):
            try:
                # 1. Lấy ảnh trang 1
                doc = fitz.open(stream=f.getvalue(), filetype="pdf")
                pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_bytes = pix.tobytes("png")
                doc.close()

                # 2. Tạo ID và tên file duy nhất
                unique_id = str(uuid.uuid4())[:8]
                new_fname = f"{unique_id}_{f.name.replace(' ', '_')}.webp"
                
                # 3. Upload lên Storage
                path = f"sketches/{new_fname}"
                supabase.storage.from_(BUCKET).upload(path, img_bytes)
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # 4. Tính Vector
                vector_data = get_vector(img_bytes)

                # 5. Ghi vào Database (ĐÃ BỎ CỘT 'specs' ĐỂ KHÔNG BÁO LỖI)
                supabase.table("ai_data").insert({
                    "file_name": str(f.name),
                    "image_url": str(img_url),
                    "vector": vector_data
                }).execute()
                
                status.success(f"✅ Đã nạp: {f.name}")
                p_bar.progress((i + 1) / len(up_new))

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
        
        st.success("🎉 Nạp kho hoàn tất!")
        st.session_state['up_key'] += 1
        time.sleep(1)
        st.rerun()




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
    st.subheader("🔄 So sánh Toàn diện (Tổng hợp tất cả Size)")

        # --- 1. CẢI TIẾN HÀM LÀM SẠCH POM (Giữ lại ý nghĩa mô tả) ---
        # 1. Hàm làm sạch POM
    def clean_pom_v4(t):
        if not t: return ""
        t = t.upper().strip()
        t = re.sub(r'^\d+[\s\.]+', '', t) 
        t = re.sub(r'[^A-Z0-9\s/]', '', t)
        return t.strip()

    # 2. Hàm xử lý số (phải để TRƯỚC khi gọi trong get_specs)
    def parse_value(text):
        if not text: return None
        text = text.replace("-", " ").strip()
        m = re.findall(r"(\d+)\s+(\d+)/(\d+)|(\d+)/(\d+)|(\d+\.?\d*)", text)
        if not m: return None
        try:
            t = m[0]
            if t[0] and t[1]: return float(t[0]) + int(t[1])/int(t[2]) # Hỗn số
            if t[3]: return int(t[3])/int(t[4]) # Phân số
            if t[5]: return float(t[5]) # Số thập phân
        except: return None
        return None

    # 3. Hàm quét PDF (Hãy đảm bảo thụt lề đúng như thế này)
    def get_specs_v5(content):
        specs_dict = {}
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df_w = pd.DataFrame(words)
                    
                    # Tìm cột Size
                    size_lanes = []
                    sz_list = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "2X", "3X"]
                    for y, gp in df_w.groupby('top'):
                        found = [w for _, w in gp.iterrows() if w['text'].strip().upper() in sz_list]
                        if len(found) >= 3:
                            for s in found:
                                size_lanes.append({"sz": s['text'].strip().upper(), "x0": s['x0']-2, "x1": s['x1']+25})
                            break 

                    if not size_lanes: continue
                    first_x = min([c['x0'] for c in size_lanes])

                    # Quét dữ liệu (Gom dòng 12px để lấy phân số nằm dưới số nguyên)
                    for _, gp in df_w.groupby(pd.cut(df_w["top"], bins=np.arange(0, page.height, 12))):
                        if gp.empty: continue
                        sorted_gp = gp.sort_values('x0')
                        pom_raw = " ".join(sorted_gp[sorted_gp['x1'] < first_x]['text']).strip()
                        pom_c = clean_pom_v4(pom_raw)
                        
                        if len(pom_c) > 3 and not any(x in pom_raw.upper() for x in ["PAGE", "STYLE"]):
                            for col in size_lanes:
                                cell = sorted_gp[(sorted_gp['x0'] >= col['x0']) & (sorted_gp['x1'] <= col['x1'])]
                                val = parse_value(" ".join(cell['text'].values))
                                if val is not None:
                                    if col['sz'] not in specs_dict: specs_dict[col['sz']] = {}
                                    specs_dict[col['sz']][pom_c] = {"orig": pom_raw, "val": val}
            return specs_dict
        except: return {}



    # --- UI GIAO DIỆN ---
    if st.button("🗑️ Làm mới hệ thống"):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A)", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B)", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ CHẠY SO SÁNH CHUẨN 100%", use_container_width=True):
            with st.spinner("Đang tách Description và bóc tách từng cột Size..."):
                dict_a = get_specs_coordinated_v3(f1.getvalue())
                dict_b = get_specs_coordinated_v3(f2.getvalue())
                if dict_a and dict_b:
                    st.session_state['ver_results'] = {"a": dict_a, "b": dict_b}
                else: st.error("❌ Không tìm thấy bảng Specs hợp lệ. Hãy kiểm tra lại file PDF.")

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        s_a, s_b = vr['a'], vr['b']
        
        # Sắp xếp Size chuyên nghiệp (2, 4, 6, 8...)
        all_sz = sorted(list(set(s_a.keys()) | set(s_b.keys())), key=lambda x: int(re.sub(r'\D', '', x)) if re.search(r'\d', x) else 99)
        tabs = st.tabs([f"Size {s}" for s in all_sz])
        
        for i, sz in enumerate(all_sz):
            with tabs[i]:
                d_a, d_b = s_a.get(sz, {}), s_b.get(sz, {})
                all_poms_c = sorted(list(set(d_a.keys()) | set(d_b.keys())))
                rows = []
                for pc in all_poms_c:
                    data_a = d_a.get(pc, {})
                    data_b = d_b.get(pc, {})
                    v1, v2 = data_a.get('val'), data_b.get('val')
                    name = data_b.get('orig') or data_a.get('orig')
                    
                    if v1 is not None and v2 is not None:
                        diff = round(v2 - v1, 3)
                        status = "✅ Khớp" if abs(diff) < 0.01 else "❌ Lệch"
                        dt_txt = f"{diff:+.3f}"
                    else:
                        dt_txt, status = "N/A", "⚠️ Thiếu dữ liệu"
                        
                    rows.append({"POM Description": name, "Bản A": v1 if v1 is not None else "-", "Bản B": v2 if v2 is not None else "-", "Lệch": dt_txt, "Kết quả": status})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=600)
