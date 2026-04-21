import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, requests, gc  # Thêm gc để dọn RAM
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from supabase import create_client

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client("YOUR_URL", "YOUR_KEY")
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (OPTIMIZED) =================
@st.cache_resource
def load_model():
    # Dùng ResNet18 nhưng ép về chế độ CPU và eval để nhẹ RAM
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

model_ai = load_model()

def get_vector(img_bytes):
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert('RGB')
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
                tensor = tf(img).unsqueeze(0)
                vec = model_ai(tensor).flatten().numpy()
                # Xóa tensor ngay sau khi tính xong
                del tensor
                
                norm = np.linalg.norm(vec)
                return (vec / norm).astype(np.float32).tolist() if norm > 0 else vec.tolist()
    except: return None
    finally:
        gc.collect() # Dọn dẹp RAM sau mỗi lần xử lý ảnh

# ================= 3. SCRAPER (GIỮ NGUYÊN) =================
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
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) # Giảm scale xuống 1.5 để nhẹ RAM
        img_bytes = pix.tobytes("png")
        doc.close()
        # Giải phóng bộ nhớ fitz
        del doc, page, pix
        gc.collect()
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR (TỐI ƯU RAM & LỌC DỮ LIỆU) =================
with st.sidebar:
    st.markdown("<h2 style='color: #1E3A8A;'>PPJ GROUP</h2>", unsafe_allow_html=True)
    
    # Hiển thị số lượng
    try:
        res_db = supabase.table("ai_data").select("id", count="exact").execute()
        current_count = res_db.count or 0
    except: current_count = 0
    
    st.metric("Models in Repo", f"{current_count} SKUs")
    
    # NÚT DỌN DẸP NÂNG CAO (Check dung lượng để xóa file trắng)
    if st.button("🧹 Dọn dẹp Database (Xóa file rác)", use_container_width=True):
        with st.spinner("Đang thanh lọc..."):
            all_data = supabase.table("ai_data").select("id, image_url").execute()
            deleted = 0
            for item in all_data.data:
                try:
                    r = requests.get(item['image_url'], timeout=5)
                    if len(r.content) < 15000: # File < 15KB thường là trắng
                        supabase.table("ai_data").delete().eq("id", item['id']).execute()
                        deleted += 1
                    del r
                except: continue
            st.success(f"Đã xóa {deleted} mẫu lỗi.")
            gc.collect()
            st.rerun()

    st.divider()
    st.subheader("📥 Nạp kho mẫu mới")
    up_new = st.file_uploader("Upload PDF", accept_multiple_files=True, key=f"up_{st.session_state['up_key']}")
    
    if up_new and st.button("🚀 ĐẨY VÀO KHO", use_container_width=True):
        p_bar = st.progress(0)
        for i, f in enumerate(up_new):
            try:
                f_bytes = f.getvalue()
                doc = fitz.open(stream=f_bytes, filetype="pdf")
                page = doc.load_page(0)
                
                # Kiểm tra ảnh (Chặn file chỉ có bảng biểu)
                if len(page.get_images()) == 0:
                    st.warning(f"Bỏ qua {f.name} (Không ảnh)")
                    doc.close()
                    continue

                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_png = pix.tobytes("png")
                
                # Upload & Vector
                unique_id = str(uuid.uuid4())[:8]
                path = f"sketches/{unique_id}_{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, img_png)
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                v_data = get_vector(img_png)
                supabase.table("ai_data").insert({"file_name": f.name, "image_url": img_url, "vector": v_data}).execute()
                
                # GIẢI PHÓNG RAM NGAY LẬP TỨC
                doc.close()
                del doc, page, pix, img_png, f_bytes
                gc.collect() 
                
                p_bar.progress((i + 1) / len(up_new))
            except Exception as e:
                st.error(f"Lỗi: {f.name}")
        
        st.session_state['up_key'] += 1
        st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
# ... (Phần UI tìm kiếm của bạn giữ nguyên, lưu ý dùng gc.collect() sau khi loop so sánh xong)





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
    st.subheader("🔄 Comprehensive Comparison")

    # --- NÚT XÓA FILE & RESET ---
    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        st.session_state.up_key += 1
        st.rerun()

    # 1. Hàm làm sạch mã POM để đối soát
    def clean_pom_universal(t):
        if not t: return ""
        t = t.upper().strip()
        match = re.search(r'([A-Z]*\d{2,4}[A-Z]*)', t)
        return match.group(1) if match else t[:25]

    # 2. Hàm xử lý số & phân số (27 1/4 -> 27.25)
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

    # 3. Hàm quét xuyên trang và lọc Header thông minh
    def get_specs_v21(content):
        specs_dict = {}
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words: continue
                    df_w = pd.DataFrame(words)
                    
                    # TÌM HEADER SIZE: Phải có từ 4 cột size trở lên và nằm vùng an toàn (x > 150)
                    size_lanes = []
                    char_sizes = ["XXS","XS","S","M","L","XL","XXL","3XL","1X","2X","3X","00","000"]
                    for y, gp in df_w.groupby('top'):
                        candidates = []
                        for _, w in gp.iterrows():
                            t = w['text'].strip().upper().replace("*", "")
                            is_char = t in char_sizes
                            is_num = t.isdigit() and (0 <= int(t) <= 60)
                            # Loại bỏ số thứ tự dòng: Nếu số nhỏ < 10 thì phải nằm bên phải (x > 200)
                            if is_num and int(t) < 10 and w['x0'] < 200: is_num = False 
                            if (is_char or is_num) and w['x0'] > 150:
                                candidates.append({"sz": t, "x0": w['x0']-12, "x1": w['x1']+25})
                        if len(candidates) >= 4:
                            size_lanes = candidates
                            break 

                    if not size_lanes: continue
                    
                    # QUÉT DỮ LIỆU CỦA TRANG CÓ BẢNG
                    first_sz_x = min([c['x0'] for c in size_lanes])
                    for _, gp in df_w.groupby(pd.cut(df_w["top"], bins=np.arange(0, page.height, 12))):
                        if gp.empty: continue
                        sorted_gp = gp.sort_values('x0')
                        pom_raw = " ".join(sorted_gp[sorted_gp['x1'] < first_sz_x]['text']).strip()
                        pom_key = clean_pom_universal(pom_raw)
                        if len(pom_key) >= 2 and not any(x in pom_raw.upper() for x in ["PAGE", "COPYRIGHT", "SPEC", "SIZE"]):
                            for col in size_lanes:
                                cell = sorted_gp[(sorted_gp['x0'] >= col['x0']) & (sorted_gp['x1'] <= col['x1'])]
                                if not cell.empty:
                                    val = parse_value_universal(" ".join(cell['text'].values))
                                    if val is not None:
                                        if col['sz'] not in specs_dict: specs_dict[col['sz']] = {}
                                        specs_dict[col['sz']][pom_key] = {"orig": pom_raw, "val": val}
            return specs_dict
        except: return {}

    # --- UI UPLOAD & SO SÁNH ---
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Old Version (A)", type="pdf", key=f"ua_{st.session_state.up_key}")
    f2 = c2.file_uploader("New Version (B)", type="pdf", key=f"ub_{st.session_state.up_key}")

    if f1 and f2:
        if st.button("⚡ RUN COMPREHENSIVE COMPARISON", use_container_width=True):
            with st.spinner("Analyzing multi-page PDF..."):
                d1, d2 = get_specs_v21(f1.getvalue()), get_specs_v21(f2.getvalue())
            
            if d1 and d2:
                # Sắp xếp size (Số trước tăng dần, chữ sau)
                def sz_rank(s):
                    if s.isdigit(): return (0, int(s))
                    return (1, s)
                all_sz = sorted(list(set(d1.keys()) | set(d2.keys())), key=sz_rank)
                all_keys = sorted(list(set([k for s in d1 for k in d1[s]]) | set([k for s in d2 for k in d2[s]])))
                
                final_rows = []
                for k in all_keys:
                    name = next((d2[s][k]['orig'] for s in d2 if k in d2[s]), next((d1[s][k]['orig'] for s in d1 if k in d1[s]), k))
                    row = {"POM Description": name}
                    for sz in all_sz:
                        v1, v2 = d1.get(sz, {}).get(k, {}).get('val'), d2.get(sz, {}).get(k, {}).get('val')
                        if v1 is not None and v2 is not None:
                            diff = round(float(v2) - float(v1), 3)
                            row[sz] = f"{v2}" if abs(diff) < 0.01 else f"{v1} ➔ {v2} [{diff:+.2f}]"
                        else: row[sz] = "-"
                    final_rows.append(row)

                df_f = pd.DataFrame(final_rows)
                
                # Nút Xuất Excel
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    df_f.to_excel(wr, index=False)
                st.download_button("📥 Download Excel Report", out.getvalue(), "Comparison_Report.xlsx")

                # Hiển thị và bôi đỏ
                st.write("### 📊 Comparison Details")
                st.dataframe(df_f.style.map(lambda x: 'background-color: #ffcccc; color: #b91c1c; font-weight: bold' if '➔' in str(x) else ''), use_container_width=True, height=600)
            else:
                st.error("❌ Spec table not found in any page. Please check the PDF content.")
