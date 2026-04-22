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

# Chuyển đổi các nhãn lựa chọn sang tiếng Anh
mode = st.radio("Select Mode:", ["Audit Mode", "Version Control"], horizontal=True)

if mode == "Audit Mode":
    st.subheader("🔍 Search Similar Models")
    
    # Đổi thông báo upload và spinner sang tiếng Anh
    target_file = st.file_uploader("Upload Target Techpack (PDF)", type=['pdf'], key=f"aud_{st.session_state['up_key']}")
    
    if target_file:
        with st.spinner("Searching for similar samples..."):
            t_data = extract_full_data(target_file.getvalue())
            
            if t_data and t_data['img']:
                t_vec = get_vector(t_data['img'])
                
                # Truy vấn dữ liệu từ Supabase
                res = supabase.table("ai_data").select("file_name, image_url, vector").execute()
                db_items = res.data
                
                if db_items and t_vec:
                    scores = []
                    for item in db_items:
                        if item['vector']:
                            # Tính toán độ tương đồng
                            sim = cosine_similarity([t_vec], [eval(item['vector']) if isinstance(item['vector'], str) else item['vector']])
                            scores.append({
                                "name": item['file_name'], 
                                "url": item['image_url'], 
                                "score": sim[0][0]
                            })
                    
                    # Lấy Top 8 mẫu giống nhất
                    top_8 = sorted(scores, key=lambda x: x['score'], reverse=True)[:8]
                    
                    st.divider()
                    st.write("### 🏆 Top Similar Matches")
                    
                    cols = st.columns(4)
                    for idx, item in enumerate(top_8):
                        with cols[idx % 4]:
                            st.image(item['url'], use_container_width=True)
                            st.caption(f"**{item['name']}**")
                            # Đổi nhãn độ giống nhau sang tiếng Anh
                            st.info(f"Similarity: {item['score']:.1%}")
                else:
                    st.warning("No data found in the database to compare.")
            else:
                st.error("Could not extract image from the uploaded PDF.")

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
