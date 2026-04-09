# ==========================================================
# AI FASHION PRO V37.9 - ZERO CRASH & DEEP POM SCAN
# ==========================================================
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- KẾT NỐI (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V37.9", page_icon="📊")

@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 nhẹ và ổn định
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- CÔNG CỤ XỬ LÝ SỐ ĐO ---
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

def extract_specs_v379(pdf_file):
    specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        # 1. Trích xuất ảnh trang 1 (Sketch)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()

        # 2. Quét POM (Ưu tiên quét Text để lấy thông số Express)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "")
                if any(k in text.upper() for k in ["POM", "SPEC", "MEASURE", "TOLERANCE"]):
                    # Cách 1: Quét bảng
                    tables = page.extract_tables()
                    for tb in tables:
                        for r in tb:
                            if r and len(r) >= 2:
                                d = str(r[0]).replace('\n',' ').strip().upper()
                                v = [parse_val(x) for x in r[1:] if parse_val(x) > 0]
                                if v: specs[d] = round(float(np.median(v)), 2)
                    
                    # Cách 2: Quét text (Dành cho dòng không kẻ ô)
                    if not specs:
                        for line in text.split('\n'):
                            nums = re.findall(r"\d+\.?\d*", line)
                            if len(nums) >= 2:
                                parts = re.split(r'\s{2,}', line.strip())
                                if len(parts) >= 2:
                                    specs[parts[0].upper()] = parse_val(nums[0])
        return {"spec": specs, "img": img_bytes}
    except: return {"spec": {}, "img": None}

# --- STYLE ---
def style_diff(val):
    return 'color: red; font-weight: bold' if abs(val) > 0.5 else 'color: white'

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_specs_v379(f)
            if d['img'] and d['spec']:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                with torch.no_grad():
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                    "img_base64": base64.b64encode(d['img']).decode()
                }, on_conflict="file_name").execute()
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Nạp xong!")

# --- MAIN ---
st.title("🔍 AI Fashion Pro V37.9")
test_file = st.file_uploader("Kéo thả file kiểm tra vào đây", type="pdf")

if test_file:
    target = extract_specs_v379(test_file)
    if target['img']:
        # Load DB
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            with torch.no_grad():
                v_test = ai_brain(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                # 🔥 FIX LỖI TYPEERROR: Kiểm tra vector trước khi tính toán
                if i.get('vector') and len(i['vector']) > 0:
                    try:
                        v_ref = np.array(i['vector']).reshape(1, -1)
                        sim = float(cosine_similarity(v_test, v_ref)) * 100
                        matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
                    except: continue
            
            if matches:
                best = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
                for m in best:
                    st.subheader(f"✨ Mẫu khớp nhất: {m['name']} ({m['sim']:.1f}%)")
                    c1, c2 = st.columns(2)
                    c1.image(target['img'], caption="Mẫu kiểm tra")
                    c2.image(base64.b64decode(m['img']), caption="Mẫu gốc")

                    # HIỂN THỊ THÔNG SỐ (TS)
                    diff = []
                    for p, v1 in target['spec'].items():
                        v2 = m['spec'].get(p, 0)
                        if v2 == 0:
                            v2 = next((val for k, val in m['spec'].items() if p[:6] in k), 0)
                        diff.append({"Hạng mục": p, "Mẫu kiểm": v1, "Mẫu gốc": v2, "Chênh lệch": round(v1-v2, 2)})
                    
                    if diff:
                        df = pd.DataFrame(diff)
                        st.table(df.style.map(style_diff, subset=['Chênh lệch']))
                        st.download_button("📥 Xuất báo cáo Excel", df.to_excel(index=False), f"Audit_{m['name']}.xlsx")
                    else:
                        st.warning("Không tìm thấy bảng thông số kỹ thuật (POM).")
            else:
                st.error("Kho dữ liệu chưa có mẫu tương thích.")
    else:
        st.error("Không trích xuất được hình ảnh từ file PDF này.")
