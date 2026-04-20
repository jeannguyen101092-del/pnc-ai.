import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
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

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (SIẾT CHẶT TÌM KIẾM) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Cắt bỏ lề nhiễu, tập trung vào Sketch giữa trang
        w, h = img.size
        img = img.crop((w*0.15, h*0.1, w*0.85, h*0.55)) 
        # Tăng tương phản để làm nổi nét phác thảo
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0).convert('RGB')

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

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.1 <= val < 150 else 0
        return 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER (FULL PAGE & NO TOL) =================
def extract_full_data(file_content):
    if not file_content:
        return None

    import fitz, pdfplumber, re, io
    import pandas as pd
    from PIL import Image

    all_specs = {}
    img_bytes = None

    try:
        # =========================
        # PREVIEW IMAGE
        # =========================
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_pil.save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue()
        doc.close()

        # =========================
        # READ PDF WORDS
        # =========================
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words:
                    continue

                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)

                size_cols = []

                # =========================
                # 1. AUTO DETECT SIZE HEADER
                # =========================
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    texts = [str(t).strip() for t in sorted_group['text']]

                    valid_tokens = []
                    for t in texts:
                        t_clean = t.lower()

                        # bỏ rác
                        if t_clean in ["grade", "tol", "+tol", "-tol", "pom"]:
                            continue

                        # match size chữ hoặc số
                        if re.match(r'^(xs|s|m|l|xl|xxl)$', t_clean) or re.match(r'^\d{1,3}$', t_clean):
                            valid_tokens.append(t)

                    # nếu >=4 size → header
                    if len(valid_tokens) >= 4:
                        for _, row in sorted_group.iterrows():
                            txt = row['text'].strip()
                            if txt in valid_tokens:
                                size_cols.append({
                                    "sz": txt.upper(),
                                    "x0": row['x0'] - 10,
                                    "x1": row['x1'] + 10
                                })
                        break

                if not size_cols:
                    continue  # không có size → skip page

                # =========================
                # 2. EXTRACT DATA
                # =========================
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text']).upper()

                    # bỏ dòng rác
                    if any(x in line_txt for x in ["COVER", "IMAGE", "DATE", "CONSTRUCTION", "MEASUREMENT"]):
                        continue

                    # 👉 lấy POM bên trái
                    left_boundary = min([c['x0'] for c in size_cols])
                    left_part = sorted_group[sorted_group['x0'] < left_boundary]

                    pom_name = " ".join(left_part['text']).strip()

                    if len(pom_name) < 3:
                        continue

                    # =========================
                    # 3. MAP VALUE
                    # =========================
                    for col in size_cols:
                        cell = sorted_group[
                            (sorted_group['x0'] >= col['x0']) &
                            (sorted_group['x1'] <= col['x1'])
                        ]

                        if not cell.empty:
                            raw = " ".join(cell['text'])

                            # parse số (có thể có fraction như 1/2)
                            val = None
                            try:
                                if "/" in raw:
                                    parts = raw.split()
                                    total = 0
                                    for p in parts:
                                        if "/" in p:
                                            num, den = p.split("/")
                                            total += float(num) / float(den)
                                        else:
                                            total += float(p)
                                    val = total
                                else:
                                    val = float(raw)
                            except:
                                continue

                            if val is not None:
                                if col['sz'] not in all_specs:
                                    all_specs[col['sz']] = {}

                                all_specs[col['sz']][pom_name] = val

        return {
            "all_specs": all_specs,
            "img": img_bytes
        }

    except Exception as e:
        print("ERROR extract:", e)
        return None

# ================= 4. SIDEBAR (DUNG LƯỢNG & TIẾN ĐỘ) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Models in Repo", f"{count} SKUs")
    
    # Hiển thị dung lượng lưu trữ
    storage_mb = count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        # Thanh tiến độ nạp kho
        prog_bar = st.progress(0)
        prog_text = st.empty()
        
        for i, f in enumerate(new_files):
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": get_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            
            # Cập nhật thanh phần trăm
            percent = (i + 1) / len(new_files)
            prog_bar.progress(percent)
            prog_text.markdown(f"**⚡ Đang xử lý:** {int(percent*100)}% ({i+1}/{len(new_files)} file)")
        
        st.success("✅ Đồng bộ hoàn tất!")
        time.sleep(1)
        st.session_state['up_key'] += 1
        st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        if target and target['img']:
            # Phân loại để ưu tiên so khớp (Áo vs Quần)
            target_name = f_audit.name.upper()
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            
            if res.data:
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                valid_rows = []
                for r in res.data:
                    if r['vector'] and len(r['vector']) == 512:
                        sim = cosine_similarity(t_vec, np.array(r['vector']).reshape(1,-1)).flatten()[0]
                        # Thưởng điểm nếu tên file cùng loại
                        if ("SHORT" in target_name and "SHORT" in r['file_name'].upper()) or \
                           ("PANT" in target_name and "PANT" in r['file_name'].upper()):
                            sim += 0.2
                        r['sim_final'] = sim
                        valid_rows.append(r)
                
                df_db = pd.DataFrame(valid_rows).sort_values('sim_final', ascending=False).head(3)
                
                st.subheader("🎯 AI Matches")
                cols = st.columns(4)
                cols[0].image(target['img'], caption="TARGET PDF", use_container_width=True)
                for i, (idx, row) in enumerate(df_db.iterrows()):
                    det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                    if det:
                        with cols[i+1]:
                            st.image(det[0]['image_url'], caption=f"Match: {min(row['sim_final'], 1.0):.1%}")
                            if st.button(f"CHỌN {i+1}", key=f"s_{idx}", use_container_width=True):
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
                        df_sz = pd.DataFrame(rows); st.table(df_sz); audit_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất Excel", to_excel(audit_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
import requests
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

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (SIẾT CHẶT TÌM KIẾM) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        # SIẾT CHẶT HƠN: Bỏ bớt phần tiêu đề phía trên và thông tin thừa hai bên
        # Cắt sâu hơn để tập trung vào hình vẽ kỹ thuật trung tâm
        img = img.crop((w*0.20, h*0.12, w*0.80, h*0.50)) 
        
        img = ImageOps.grayscale(img)
        # Tăng tương phản cao hơn nữa để tách biệt hẳn nét vẽ và nền trắng
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


def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.1 <= val < 150 else 0
        return 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER (FULL PAGE & NO TOL) =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    
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
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "DATE", "CONSTRUCTION"]): continue
                    
                    pom_name = re.sub(r'[\d./\s]+$', '', " ".join(sorted_group[sorted_group['x1'] < 350]['text'])).strip()
                    if len(pom_name) > 3:
                        for col in size_cols:
                            cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR (CHỈNH SỬA PHẦN NÂNG CẤP) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        count = res_count.count or 0
    except:
        count = 0
        
    st.metric("Models in Repo", f"{count} SKUs")
    
    storage_mb = count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()

    # --- ĐOẠN NÂNG CẤP ĐÃ ĐƯỢC FIX LỖI THỤT LỀ ---
    st.subheader("⚙️ Bảo trì hệ thống")
    with st.expander("Nâng cấp kho dữ liệu AI"):
        st.info("Quét lại 1.390+ mẫu để nhận diện Quần/Áo chính xác hơn.")
        if st.button("🚀 BẮT ĐẦU NÂNG CẤP", use_container_width=True):
            try:
                res = supabase.table("ai_data").select("id, image_url").execute()
                items = res.data
                if items:
                    prog_bar = st.progress(0)
                    status_txt = st.empty()
                    for i, item in enumerate(items):
                        try:
                            resp = requests.get(item['image_url'], timeout=10)
                            if resp.status_code == 200:
                                new_vec = get_vector(resp.content)
                                if new_vec:
                                    supabase.table("ai_data").update({"vector": new_vec}).eq("id", item['id']).execute()
                            
                            percent = (i + 1) / len(items)
                            prog_bar.progress(percent)
                            status_txt.markdown(f"**⏳ Tiến độ:** {i+1}/{len(items)}")
                        except: 
                            continue
                    st.success("✅ Hoàn tất nâng cấp!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        # Phần code xử lý Synchronize của bạn tiếp tục ở đây...
        st.write("Đang đồng bộ...")
       # Thanh tiến độ nạp kho
        prog_bar = st.progress(0)
        prog_text = st.empty()
        
        for i, f in enumerate(new_files):
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": get_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            
            # Cập nhật thanh phần trăm
            percent = (i + 1) / len(new_files)
            prog_bar.progress(percent)
            prog_text.markdown(f"**⚡ Đang xử lý:** {int(percent*100)}% ({i+1}/{len(new_files)} file)")
        
        st.success("✅ Đồng bộ hoàn tất!")
        time.sleep(1)
        st.session_state['up_key'] += 1
        st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")

# Dùng tên đơn giản để tránh lỗi so sánh chuỗi
mode = st.radio("Chế độ:", ["Audit Mode", "Version Control"], horizontal=True)

if mode == "Audit Mode":
    st.subheader("🔍 Tìm kiếm mẫu tương đồng & Audit")
    # Thêm nút xóa nhanh để reset
    if st.button("🗑️ Làm mới tìm kiếm"):
        st.session_state['up_key'] += 1
        st.rerun()
        
    up_target = st.file_uploader("Upload Target PDF", type=['pdf'], key=f"target_{st.session_state['up_key']}")
    if up_target:
        st.info(f"Đã nhận file: {up_target.name}. Đang xử lý phân tích...")
        # Logic Audit của bạn tiếp tục ở đây...

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        
        if target and target.get('img'):
            # --- 1. HIỂN THỊ MẪU GỐC ---
            st.image(target['img'], width=280, caption="Mẫu bạn đang kiểm tra")
            st.divider()

            # 2. XÁC ĐỊNH CHỦNG LOẠI CỦA FILE UPLOAD (TARGET)
            t_name = f_audit.name.upper()
            is_t_bottom = any(x in t_name for x in ["PANT", "SHORT", "TROUSER", "JEAN", "LEG"])
            is_t_top = any(x in t_name for x in ["SHIRT", "JACKET", "TOP", "TEE", "POLO", "HOODIE"])

            with st.spinner("🕵️ AI đang đối soát và lọc chủng loại..."):
                res = supabase.table("ai_data").select("id, vector, file_name, image_url").execute()
                
                if res.data:
                    t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                    valid_rows = []
                    
                    for r in res.data:
                        r_name = r['file_name'].upper()
                        # XÁC ĐỊNH CHỦNG LOẠI TRONG DATABASE
                        is_r_bottom = any(x in r_name for x in ["PANT", "SHORT", "TROUSER", "JEAN"])
                        is_r_top = any(x in r_name for x in ["SHIRT", "JACKET", "TOP", "TEE"])

                        # --- BỘ LỌC CỨNG (HARD FILTER) ---
                        # Nếu cùng là Quần hoặc cùng là Áo thì mới tính tiếp
                        if (is_t_bottom == is_r_bottom) or (is_t_top == is_r_top):
                            sim_img = cosine_similarity(t_vec, np.array(r['vector']).reshape(1,-1)).flatten()[0]
                            
                            # Thưởng điểm nếu khớp chính xác từ khóa (Vd: Cùng là SHORT)
                            if ("SHORT" in t_name and "SHORT" in r_name) or ("PANT" in t_name and "PANT" in r_name):
                                sim_img += 0.15
                            
                            r['sim_final'] = min(sim_img, 1.0)
                            valid_rows.append(r)
                    
                    # 3. HIỂN THỊ TOP 8 (CHỈ HIỆN CÙNG LOẠI)
                    if not valid_rows:
                        st.warning("⚠️ Không tìm thấy mẫu nào cùng chủng loại (Quần/Áo) trong Database.")
                    else:
                        df_db = pd.DataFrame(valid_rows).sort_values('sim_final', ascending=False).head(8)
                        st.subheader(f"🎯 Top 8 mẫu {'QUẦN' if is_t_bottom else 'ÁO' if is_t_top else ''} tương đồng nhất")
                        
                        for r_idx in range(2):
                            cols = st.columns(4)
                            for c_idx in range(4):
                                idx = r_idx * 4 + c_idx
                                if idx < len(df_db):
                                    item = df_db.iloc[idx]
                                    with cols[c_idx]:
                                        st.image(item['image_url'], use_container_width=True)
                                        st.write(f"**Chính xác: {item['sim_final']:.1%}**")
                                        st.caption(f"📄 {item['file_name'][:20]}...")
                                        if st.button("Chọn mẫu này", key=f"btn_{idx}"):
                                            st.session_state['sel_audit'] = item.to_dict()
                                            st.rerun()
elif mode == "Version Control":
    st.subheader("🔄 So sánh 2 file PDF (ALL SIZE)")

    # Nút xóa để reset uploader
    if st.button("🗑️ Xoá file đã upload", use_container_width=True):
        st.session_state['up_key'] += 1         
        st.session_state['ver_results'] = None  
        st.rerun()

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key=f"v2_{st.session_state['up_key']}")

    # =========================
    # RUN COMPARE
    # =========================
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh toàn diện", use_container_width=True):
            with st.spinner("Đang quét toàn bộ dữ liệu..."):
                d1 = extract_full_data(f1.getvalue())
                d2 = extract_full_data(f2.getvalue())

                if d1 and d2 and d1.get('all_specs') and d2.get('all_specs'):
                    st.session_state['ver_results'] = {
                        "d1": d1, "d2": d2,
                        "f1_name": f1.name, "f2_name": f2.name
                    }
                else:
                    st.error("❌ Không đọc được bảng thông số từ một trong hai file.")

    # =========================
    # SHOW RESULT
    # =========================
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider()

        # Hiển thị ảnh minh họa
        col_img_a, col_img_b = st.columns(2)
        col_img_a.image(vr['d1']['img'], caption=f"Bản A: {vr['f1_name']}", use_container_width=True)
        col_img_b.image(vr['d2']['img'], caption=f"Bản B: {vr['f2_name']}", use_container_width=True)

        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        version_dfs, ver_sheets = [], []

        def color_status(val):
            if val == "❌ Lệch": return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
            if val == "✅ Khớp": return 'background-color: #ccffcc; color: #006600;'
            return 'background-color: #fff3cd; color: #856404;'

        for sz in all_sz:
            with st.expander(f"📊 CHI TIẾT SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                all_poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []

                for p in all_poms:
                    v1, v2 = s1.get(p), s2.get(p)
                    if v1 is not None and v2 is not None:
                        diff_val = round(v2 - v1, 3)
                        status = "✅ Khớp" if abs(diff_val) < 0.001 else "❌ Lệch"
                        diff_txt = f"{diff_val:+.3f}"
                    else:
                        diff_txt, status = "N/A", "⚠️ Thiếu"

                    rows.append({"POM": p, "Bản A": v1, "Bản B": v2, "Chênh lệch": diff_txt, "Kết quả": status})

                df_sz = pd.DataFrame(rows)
                try:
                    st.dataframe(df_sz.style.map(color_status, subset=['Kết quả']), use_container_width=True)
                except:
                    st.dataframe(df_sz.style.applymap(color_status, subset=['Kết quả']), use_container_width=True)
                
                version_dfs.append(df_sz)
                ver_sheets.append(f"Size_{sz}")

        st.divider()
        if version_dfs:
            col_exp, col_reset = st.columns(2)
            with col_exp:
                st.download_button("📥 Tải Excel So Sánh", to_excel(version_dfs, ver_sheets), "Comparison.xlsx", use_container_width=True)
            with col_reset:
                if st.button("🗑️ Xóa & Làm lại", use_container_width=True):
                    st.session_state['ver_results'] = None
                    st.session_state['up_key'] += 1
                    st.rerun()
