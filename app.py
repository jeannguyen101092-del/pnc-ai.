# ==========================================================
# AI FASHION AUDITOR V38.2 - ZERO CRASH & STABLE COMPARISON
# ==========================================================
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V38.2", page_icon="📊")

@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 ổn định cho nhận diện form dáng
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- CÔNG CỤ XỬ LÝ ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip()
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def advanced_classify(text, file_name):
    txt = (str(text) + " " + str(file_name)).upper()
    if any(k in txt for k in ['INSEAM', 'RISE', 'PANT', 'TROUSER', 'SHORT', 'JEAN']):
        return "QUẦN"
    if any(k in txt for k in ['DRESS', 'SKIRT', 'VAY', 'DAM']):
        return "VÁY/ĐẦM"
    return "ÁO"

def extract_data(pdf_file):
    specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").upper()
                all_txt += txt + " "
                if any(k in txt for k in ["POM", "SPEC", "MEASURE"]):
                    tables = page.extract_tables()
                    for tb in tables:
                        for r in tb:
                            if r and len(r) >= 2:
                                desc = str(r[0]).replace('\n',' ').strip().upper()
                                if len(desc) < 3 or any(x in desc for x in ["DATE","PAGE","SIZE"]): continue
                                v_list = [parse_val(x) for x in r[1:] if parse_val(x) > 0]
                                if v_list: specs[desc] = round(float(np.median(v_list)), 2)
        
        cat = advanced_classify(all_txt, pdf_file.name)
        return {"spec": specs, "img": img_bytes, "cat": cat}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Số mẫu trong kho", res_db.count if res_db.count else 0)
    
    files = st.file_uploader("Nạp Techpacks (PDF)", accept_multiple_files=True)
    if files and st.button("🚀 NẠP VÀO HỆ THỐNG"):
        p_bar = st.progress(0)
        p_text = st.empty()
        for i, f in enumerate(files):
            d = extract_data(f)
            if d and d['img']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                try:
                    supabase.storage.from_(BUCKET).upload(path=f"{f.name}.png", file=d['img'], file_options={"content-type": "image/png", "x-upsert": "true"})
                except: pass

                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                    "img_base64": base64.b64encode(d['img']).decode(), "category": d['cat']
                }, on_conflict="file_name").execute()
            
            pct = (i + 1) / len(files)
            p_bar.progress(pct)
            p_text.text(f"Đang nạp: {int(pct*100)}%")
        st.success("✅ Đã nạp xong!")
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V38.2")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = extract_data(test_file)
    if target and target['img']:
        st.info(f"Nhận diện: **{target['cat']}** | Tìm thấy {len(target['spec'])} hạng mục POM")
        
        # Lọc theo loại hàng (Áo/Quần)
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                if i.get('vector'):
                    try:
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        # FIX LỖI TYPEERROR TẠI ĐÂY
                        similarity_matrix = cosine_similarity(v_test, v_ref)
                        sim = float(similarity_matrix[0][0]) * 100
                        matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
                    except: continue
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in best:
                st.subheader(f"✨ Mẫu khớp nhất: {m['name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                c1.image(target['img'], caption="File đang kiểm")
                c2.image(base64.b64decode(m['img']), caption="Mẫu trong kho")

                diff = []
                for p, v1 in target['spec'].items():
                    v2 = m['spec'].get(p, 0)
                    if v2 == 0: # Tìm từ khóa tương đối
                        v2 = next((val for k, val in m['spec'].items() if p[:6] in k), 0)
                    diff.append({"Hạng mục": p, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": round(v1-v2, 2)})
                
                if diff:
                    df = pd.DataFrame(diff)
                    # Tự động bôi đỏ dòng lệch > 0.5
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
        else:
            st.warning(f"⚠️ Kho chưa có mẫu nào thuộc loại '{target['cat']}'.")
