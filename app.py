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
        w, h = img.size
        img = img.crop((w*0.15, h*0.1, w*0.85, h*0.55)) 
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
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num: return float(num[0])
        return 0
    except: return 0

# ================= 3. SCRAPER =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
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
                # ... (Phần logic tìm size và POM giữ nguyên)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR (FIXED INDENTATION) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    
    # Đoạn hiển thị Metric
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

    st.subheader("⚙️ Bảo trì hệ thống")
    with st.expander("Nâng cấp kho dữ liệu AI"):
        st.info("Quét lại 1.390+ mẫu để nhận diện Quần/Áo chính xác hơn.")
        if st.button("🚀 BẮT ĐẦU NÂNG CẤP", use_container_width=True):
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
                    except: continue
                st.success("✅ Hoàn tất nâng cấp!")
                st.balloons()
                time.sleep(2)
                st.rerun()

    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")

    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
                # --- DÁN ĐOẠN NÀY VÀO DÒNG 155 ---
               # --- ĐOẠN NÂNG CẤP CHUẨN ---
            st.subheader("⚙️ Bảo trì hệ thống")
        with st.expander("Nâng cấp kho dữ liệu AI"):
            st.info("Quét lại 1.390+ mẫu để nhận diện Quần/Áo chính xác hơn.")
            if st.button("🚀 BẮT ĐẦU NÂNG CẤP", use_container_width=True):
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
                        except: continue
                    st.success("✅ Hoàn tất nâng cấp!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
        st.divider()

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



elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh 2 file PDF (ALL PAGE + ALL SIZE)")

    # --- NÚT XÓA FILE (Đưa lên đầu để reset trạng thái) ---
    if st.button("🗑️ Xoá file đã upload", use_container_width=True):
        st.session_state['up_key'] += 1         # Thay đổi key để reset widget uploader
        st.session_state['ver_results'] = None  # Xóa kết quả so sánh cũ
        st.rerun()

    c1, c2 = st.columns(2)
    # Thêm biến up_key vào key để Streamlit nhận diện reset khi bấm nút
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

                # DEBUG SIZE
                size_a = list(d1['all_specs'].keys()) if d1 and d1.get('all_specs') else []
                size_b = list(d2['all_specs'].keys()) if d2 and d2.get('all_specs') else []

                st.write("📊 SIZE A:", size_a)
                st.write("📊 SIZE B:", size_b)

                # CHECK DATA
                if not size_a:
                    st.error("❌ File A không đọc được bảng thông số")
                    st.stop()

                if not size_b:
                    st.error("❌ File B không đọc được bảng thông số")
                    st.stop()

                st.session_state['ver_results'] = {
                    "d1": d1,
                    "d2": d2,
                    "f1_name": f1.name,
                    "f2_name": f2.name
                }

    # =========================
    # SHOW RESULT
    # =========================
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']

        st.divider()

        col_a, col_b = st.columns(2)
        col_a.image(vr['d1']['img'], caption="Bản A", use_container_width=True)
        col_b.image(vr['d2']['img'], caption="Bản B", use_container_width=True)

        # lấy tất cả size
        all_sz = sorted(
            list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())),
            key=lambda x: str(x)
        )

        if not all_sz:
            st.warning("⚠️ Không tìm thấy SIZE nào để so sánh")
            st.stop()

        version_dfs = []
        ver_sheets = []

        # =========================
        # LOOP SIZE
        # =========================
        for sz in all_sz:
            with st.expander(f"SIZE: {sz}", expanded=True):

                s1 = vr['d1']['all_specs'].get(sz, {})
                s2 = vr['d2']['all_specs'].get(sz, {})

                if not s1 and not s2:
                    st.warning(f"⚠️ SIZE {sz} không có dữ liệu")
                    continue

                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []

                for p in poms:
                    v1 = s1.get(p)
                    v2 = s2.get(p)

                    if v1 is None or v2 is None:
                        diff = "N/A"
                        status = "⚠️ Missing"
                    else:
                        diff_val = v2 - v1
                        diff = f"{diff_val:+.3f}"
                        status = "✅" if abs(diff_val) < 1e-6 else "⚠️"

                    rows.append({
                        "Point": p,
                        "Ver A": v1,
                        "Ver B": v2,
                        "Diff": diff,
                        "Status": status
                    })

                if rows:
                    df_sz = pd.DataFrame(rows)
                    st.dataframe(df_sz, use_container_width=True)

                    version_dfs.append(df_sz)
                    ver_sheets.append(f"Size_{sz}")

        # =========================
        # EXPORT EXCEL
        # =========================
        if version_dfs:
            st.download_button(
                "📥 Xuất Excel So Sánh",
                to_excel(version_dfs, ver_sheets),
                "Comparison.xlsx",
                use_container_width=True
            )
