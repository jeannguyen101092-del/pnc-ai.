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
                    # Bỏ qua các dòng tiêu đề hoặc rác
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

elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh & Đối chiếu Thông số Kỹ thuật")
    
    # 1. Tải lên 2 file để đối chiếu
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Chọn File A (Gốc):", type="pdf", key="v1")
    f2 = c2.file_uploader("Chọn File B (Mới):", type="pdf", key="v2")

    if f1 and f2:
        if st.button("🚀 Bắt đầu truy quét toàn bộ các trang", use_container_width=True):
            with st.spinner("🔍 Đang đọc dữ liệu từ trang 1 đến trang cuối..."):
                # Gọi hàm xử lý (Đảm bảo hàm này duyệt qua page trong pdf.pages)
                d1 = extract_full_data(f1.getvalue())
                d2 = extract_full_data(f2.getvalue())
                
                if d1 and d2:
                    st.session_state['ver_results'] = {
                        "d1": d1, "d2": d2, 
                        "f1_name": f1.name, "f2_name": f2.name
                    }

    # 2. Hiển thị kết quả so sánh nếu có dữ liệu
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        
        # Hiển thị ảnh các trang để kiểm chứng hệ thống đã đọc đến trang 13 chưa
        with st.expander("🖼️ Xem lại các trang PDF đã quét dữ liệu"):
            t1, t2 = st.tabs([f"Trang trong {vr['f1_name']}", f"Trang trong {vr['f2_name']}"])
            with t1:
                imgs_a = vr['d1'].get('imgs', [])
                st.image(imgs_a, caption=[f"Trang {i+1}" for i in range(len(imgs_a))], width=280)
            with t2:
                imgs_b = vr['d2'].get('imgs', [])
                st.image(imgs_b, caption=[f"Trang {i+1}" for i in range(len(imgs_b))], width=280)

        # Gom tất cả các Size tìm thấy trên mọi trang của cả 2 file
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        
        version_dfs, ver_sheets = [], []
        st.write(f"### 📊 Bảng so sánh chi tiết ({len(all_sz)} Size được tìm thấy)")

        if not all_sz:
            st.error("❌ Không tìm thấy bảng thông số POM. Hãy kiểm tra lại cấu trúc file.")
        
        for sz in all_sz:
            with st.expander(f"📏 CHI TIẾT SIZE: {sz}", expanded=True):
                s1 = vr['d1']['all_specs'].get(sz, {})
                s2 = vr['d2']['all_specs'].get(sz, {})
                
                # Lấy tất cả các điểm đo (POM) tìm thấy trên mọi trang
                all_poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                
                rows = []
                for p in all_poms:
                    v1, v2 = s1.get(p, 0), s2.get(p, 0)
                    
                    # Tính toán chênh lệch (xử lý cả trường hợp số phân số)
                    try:
                        diff = float(v2) - float(v1)
                        diff_str = f"{diff:+.3f}" if diff != 0 else "0"
                    except:
                        diff_str = "N/A"

                    # Phân loại trạng thái để tô màu
                    if v1 == v2: 
                        status = "✅ Khớp"
                    elif v1 != 0 and (v2 == 0 or v2 is None): 
                        status = "❌ File B bị mất"
                    elif (v1 == 0 or v1 is None) and v2 != 0: 
                        status = "➕ File B thêm mới"
                    else: 
                        status = "⚠️ Lệch"

                    rows.append({
                        "Mã & Mô tả POM": p,
                        "File A": v1,
                        "File B": v2,
                        "Chênh lệch": diff_str,
                        "Trạng thái": status
                    })

                df_sz = pd.DataFrame(rows)
                
                # Style để làm nổi bật lỗi chênh lệch
                def highlight_diff(row):
                    color = ''
                    if "⚠️" in row['Trạng thái']: color = 'background-color: #fff3cd' # Vàng
                    elif "❌" in row['Trạng thái']: color = 'background-color: #f8d7da' # Đỏ
                    elif "➕" in row['Trạng thái']: color = 'background-color: #d1ecf1' # Xanh dương
                    return [color] * len(row)

                st.dataframe(df_sz.style.apply(highlight_diff, axis=1), use_container_width=True)
                version_dfs.append(df_sz); ver_sheets.append(str(sz))

        # 3. Xuất file Excel tổng hợp mọi trang
        st.divider()
        st.download_button(
            label="📥 Tải báo cáo so sánh đầy đủ (.xlsx)",
            data=to_excel(version_dfs, ver_sheets),
            file_name=f"Full_Comparison_Report.xlsx",
            use_container_width=True
        )
