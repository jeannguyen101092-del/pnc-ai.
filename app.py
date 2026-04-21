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
    st.subheader("🔄 So sánh Toàn diện (Bản vạn năng - Đã fix nhiễu Express)")

    def clean_pom_universal(t):
        if not t: return ""
        t = t.upper().strip()
        match = re.search(r'([A-Z]*\d{2,4}[A-Z]*)', t)
        if match: return match.group(1)
        return re.sub(r'[^A-Z0-9]', '', t)

    def parse_value_universal(text):
        if not text: return None
        text = text.replace("-", " ").strip()
        m = re.findall(r"(\d+)\s+(\d+)/(\d+)|(\d+)/(\d+)|(\d+\.?\d*)", text)
        if not m: return None
        try:
            t = m[0]
            if t[0] and t[1]: return float(t[0]) + int(t[1])/int(t[2])
            if t[3]: return int(t[3])/int(t[4])
            if t[5]: return float(t[5])
        except: return None
        return None

    def get_specs_universal(content):
        specs_dict = {}
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df_w = pd.DataFrame(words)
                    
                    # --- BỘ LỌC SIZE CHUẨN (Fix lỗi bốc nhầm MENS, ALL, 2024...) ---
                    size_lanes = []
                    # Danh sách Size "vạn năng" bao quát mọi khách hàng
                    valid_patterns = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "1X", "2X", "3X", "000", "00"]
                    
                    for y, gp in df_w.groupby('top'):
                        candidates = []
                        for _, w in gp.iterrows():
                            t = w['text'].strip().upper().replace("*", "")
                            # CHỈ CHẤP NHẬN: 1. Có trong list valid HOẶC 2. Là số nguyên từ 24-56
                            is_size_num = t.isdigit() and (24 <= int(t) <= 56 or t in ["0", "2", "4", "6", "8"])
                            is_size_char = t in valid_patterns
                            
                            if (is_size_num or is_size_char) and w['x0'] > 180:
                                candidates.append({"sz": t, "x0": w['x0']-12, "x1": w['x1']+28})
                        
                        if len(candidates) >= 4:
                            size_lanes = candidates
                            break 

                    if not size_lanes: continue
                    first_x = min([c['x0'] for c in size_lanes])

                    for _, gp in df_w.groupby(pd.cut(df_w["top"], bins=np.arange(0, page.height, 12))):
                        if gp.empty: continue
                        sorted_gp = gp.sort_values('x0')
                        pom_raw = " ".join(sorted_gp[sorted_gp['x1'] < first_x]['text']).strip()
                        pom_key = clean_pom_universal(pom_raw)
                        
                        # Chỉ lấy dòng có mã POM (chứa số) để bỏ qua các dòng tiêu đề rác
                        if re.search(r'\d', pom_key) and not any(x in pom_raw.upper() for x in ["COPYRIGHT", "PRINTED", "PAGE"]):
                            for col in size_lanes:
                                cell = sorted_gp[(sorted_gp['x0'] >= col['x0']) & (sorted_gp['x1'] <= col['x1'])]
                                val = parse_value_universal(" ".join(cell['text'].values))
                                if val is not None:
                                    if col['sz'] not in specs_dict: specs_dict[col['sz']] = {}
                                    specs_dict[col['sz']][pom_key] = {"orig": pom_raw, "val": val}
            return specs_dict
        except: return {}

    # --- UI & XUẤT FILE ---
    f1 = st.file_uploader("Bản cũ (A)", type="pdf", key="u_a")
    f2 = st.file_uploader("Bản mới (B)", type="pdf", key="u_b")

    if f1 and f2:
        if st.button("⚡ CHẠY SO SÁNH TỔNG LỰC", use_container_width=True):
            d1, d2 = get_specs_universal(f1.getvalue()), get_specs_universal(f2.getvalue())
            if d1 and d2:
                all_sz = sorted(list(set(d1.keys()) | set(d2.keys())), key=lambda x: str(x).zfill(3))
                all_keys = sorted(list(set([k for s in d1 for k in d1[s]]) | set([k for s in d2 for k in d2[s]])))
                
                final_rows = []
                for k in all_keys:
                    name = next((d2[s][k]['orig'] for s in d2 if k in d2[s]), next((d1[s][k]['orig'] for s in d1 if k in d1[s]), k))
                    row = {"POM Description": name}
                    for sz in all_sz:
                        v1, v2 = d1.get(sz, {}).get(k, {}).get('val'), d2.get(sz, {}).get(k, {}).get('val')
                        if v1 is not None and v2 is not None:
                            diff = round(v2 - v1, 3)
                            row[sz] = f"{v2}" if abs(diff) < 0.01 else f"{v1} ➔ {v2} [{diff:+.2f}]"
                        else: row[sz] = "-"
                    final_rows.append(row)

                df_final = pd.DataFrame(final_rows)
                
                # Nút xuất Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_final.to_excel(writer, index=False)
                st.download_button("📥 Tải báo cáo Excel", output.getvalue(), "Bao_cao_so_sanh.xlsx")

                # Hiển thị bôi đỏ
                st.dataframe(df_final.style.map(lambda x: 'background-color: #ffcccc; color: #b91c1c; font-weight: bold' if '➔' in str(x) else ''), use_container_width=True, height=600)
            else: st.error("❌ Không tìm thấy bảng dữ liệu chuẩn.")
