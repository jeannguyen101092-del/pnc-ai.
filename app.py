import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, requests, gc
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

# ================= 2. AI CORE (OPTIMIZED RAM) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

model_ai = load_model()

def get_vector(img_bytes):
    try:
        with Image.open(io.BytesIO(img_bytes)) as img_org:
            img = img_org.convert('RGB')
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
                input_tensor = tf(img).unsqueeze(0)
                vec = model_ai(input_tensor).flatten().numpy()
                del input_tensor
                norm = np.linalg.norm(vec)
                return (vec / norm).astype(np.float32).tolist() if norm > 0 else vec.tolist()
    except: return None
    finally: gc.collect()

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
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        # Giảm Matrix xuống 1.5 để tiết kiệm RAM
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()
        del doc, page, pix
        gc.collect()
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR (SMART CLEANUP & UPLOAD) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    
    try:
        res_db = supabase.table("ai_data").select("id", count="exact").execute()
        current_count = res_db.count or 0
    except: current_count = 0
    
    st.metric("Models in Repo", f"{current_count} SKUs")
    storage_mb = current_count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    
    # --- DỌN DẸP THÔNG MINH (XÓA FILE HOLLISTER CHỈ CÓ CHỮ) ---
    if st.button("🧹 Dọn dẹp Database (Lọc sâu)", use_container_width=True):
        with st.spinner("Đang phân tích và xóa file rác..."):
            all_data = supabase.table("ai_data").select("id, image_url").execute()
            deleted = 0
            for item in all_data.data:
                try:
                    r = requests.get(item['image_url'], timeout=5)
                    with Image.open(io.BytesIO(r.content)) as img:
                        # Kiểm tra độ lệch chuẩn để biết là trang trắng/bảng biểu hay sketch
                        std_val = np.array(img.convert('L')).std()
                        # Nếu std < 12 hoặc file < 25KB -> Khả năng cao là bảng biểu
                        if std_val < 12 or len(r.content) < 25000:
                            supabase.table("ai_data").delete().eq("id", item['id']).execute()
                            deleted += 1
                    del r
                except: continue
            st.success(f"🔥 Đã dọn dẹp {deleted} file không đạt chuẩn!")
            gc.collect()
            st.rerun()

    if st.button("🔄 Làm mới số lượng"): st.rerun()

    st.divider()
    st.subheader("📥 Nạp kho mẫu mới")
    up_new = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"side_up_{st.session_state['up_key']}")
    
    if up_new and st.button("🚀 ĐẨY VÀO KHO", use_container_width=True):
        p_bar = st.progress(0)
        for i, f in enumerate(up_new):
            try:
                f_bytes = f.getvalue()
                doc = fitz.open(stream=f_bytes, filetype="pdf")
                page = doc.load_page(0)
                
                # Lấy ảnh preview
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_png = pix.tobytes("png")
                
                # LỌC NGAY LÚC UPLOAD: Kiểm tra độ lệch chuẩn của trang
                with Image.open(io.BytesIO(img_png)) as tmp_img:
                    std_val = np.array(tmp_img.convert('L')).std()
                
                if std_val < 12 or len(img_png) < 25000:
                    st.warning(f"⏩ Bỏ qua {f.name}: Chỉ có bảng biểu.")
                    doc.close()
                    continue

                unique_id = str(uuid.uuid4())[:8]
                new_fname = f"{unique_id}_{f.name.replace(' ', '_')}.webp"
                path = f"sketches/{new_fname}"
                
                supabase.storage.from_(BUCKET).upload(path, img_png)
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                v_data = get_vector(img_png)
                supabase.table("ai_data").insert({
                    "file_name": str(f.name),
                    "image_url": str(img_url),
                    "vector": v_data
                }).execute()
                
                doc.close()
                del doc, page, pix, img_png, f_bytes
                gc.collect()
                p_bar.progress((i + 1) / len(up_new))
            except Exception as e:
                st.error(f"❌ Lỗi: {f.name}")
        
        st.success("🎉 Hoàn tất!")
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
    st.subheader("🔄 Comprehensive Comparison")

    # --- NÚT XÓA FILE & RESET ---
    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        st.session_state.up_key += 1
        st.rerun()

    # 1. Hàm làm sạch mã POM để đối soát
