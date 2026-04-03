# ==========================================================
# AI FASHION PRO V8.1 - STRICT FILTERING & CLASSIFICATION
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.1", page_icon="👔")

# ================= AI MODEL =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ================= CÔNG CỤ XỬ LÝ DỮ LIỆU =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE','SHOULDER','BOTTOM']
BLOCK_KEYS = ['SIZE','SEASON','TECH','DATE','#','DEVELOPMENT','FABRIC','BODY','SHELL','LINING','MATERIAL','%','PFD','DYED','WASH','COLOR','PRINT']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r or len(r) < 2: continue
        row_text = " | ".join([str(x) for x in r if x]).upper()
        if any(x in row_text for x in BLOCK_KEYS): continue
        if not any(k in row_text for k in VALID_KEYS): continue
        vals = [parse_val(x) for x in r if x and parse_val(x) > 0]
        if not vals: continue
        val = float(np.median(vals))
        specs[row_text[:100]] = round(val, 2)
    return specs

def advanced_classify(specs, text, file_name):
    txt = (text + " " + file_name).upper()
    inseam = next((v for k, v in specs.items() if 'INSEAM' in k), 0)
    
    if 'BIB' in txt: return "QUẦN YẾM"
    if 'CARGO' in txt: return "QUẦN CARGO"
    if inseam > 0:
        if inseam <= 11: return "QUẦN SHORT"
        if inseam >= 25: return "QUẦN DÀI"
        return "QUẦN LỬNG"
    if 'DRESS' in txt: return "ĐẦM"
    if 'SKIRT' in txt: return "VÁY"
    if 'SHIRT' in txt: return "ÁO SƠ MI"
    return "ÁO"

# ================= HÀM ĐỌC PDF & KIỂM TRA ĐIỀU KIỆN =================
def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        # 1. Trích xuất thông số
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: all_texts += t + " "
                for table in p.extract_tables():
                    specs.update(extract_specs(table))
        
        # KIỂM TRA ĐIỀU KIỆN 1: Phải có ít nhất 5 dòng thông số kỹ thuật
        if len(specs) < 5:
            return {"error": "Thiếu thông số (ít hơn 5 dòng POM)"}

        # 2. Trích xuất hình ảnh
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        
        # KIỂM TRA ĐIỀU KIỆN 2: Hình ảnh phải trích xuất thành công
        if not img_bytes or len(img_bytes) < 1000:
            return {"error": "Không tìm thấy hình ảnh minh họa hợp lệ"}

        img_b64 = base64.b64encode(img_bytes).decode()
        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))
        
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": cat}
    except Exception as e:
        return {"error": str(e)}

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    
    # Hiển thị tổng kho
    try:
        count_res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{count_res.count} mẫu")
    except: st.metric("Tổng mẫu trong kho", "0 mẫu")

    if "up_key" not in st.session_state: st.session_state.up_key = 0
    files = st.file_uploader("Upload PDF nạp kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")

    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        status = st.empty()
        success_count = 0

        for idx, f in enumerate(files):
            progress = (idx + 1) / len(files)
            p_bar.progress(progress)
            status.text(f"Đang kiểm tra: {f.name} ({int(progress*100)}%)")

            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")

            # NẾU FILE KHÔNG ĐỦ ĐIỀU KIỆN -> BỎ QUA
            if "error" in d:
                st.warning(f"⚠️ Bỏ qua {f.name}: {d['error']}")
                continue

            # Xử lý AI Vector
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                img_obj = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                vec = ai_brain(tf(img_obj).unsqueeze(0)).flatten().numpy().tolist()

            # Lưu vào database
            supabase.table("ai_data").upsert({
                "file_name": re.sub(r'\s*\(\d+\)', '', f.name),
                "vector": vec,
                "spec_json": d['spec'],
                "img_base64": d['img_b64'],
                "category": d['cat']
            }, on_conflict="file_name").execute()
            success_count += 1
            os.remove("tmp.pdf")

        st.session_state.up_key += 1
        st.success(f"✅ Đã nạp thành công {success_count}/{len(files)} file đủ điều kiện!")
        st.rerun()

# ================= PHẦN SO SÁNH CHÍNH =================
st.title("👔 AI Fashion Pro V8.1")
test_file = st.file_uploader("Tải file PDF cần đối chứng", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")

    if "error" in target:
        st.error(f"❌ File test không hợp lệ: {target['error']}")
    else:
        st.info(f"Phân loại mẫu: **{target['cat']}**")
        
        # CHỈ LẤY CÁC MẪU CÙNG LOẠI TỪ KHO (Ví dụ: Quần dài chỉ tìm Quần dài)
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            results = []
            for i in db.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    results.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})

            results = sorted(results, key=lambda x: x['sim'], reverse=True)[:10]

            for r in results:
                with st.expander(f"🎯 {r['sim']:.1f}% | {r['name']}"):
                    c1, c2 = st.columns(2)
                    with c1: st.image(target['img_bytes'], caption="Mẫu Test")
                    with c2: st.image(base64.b64decode(r['img']), caption="Mẫu Kho")

                    diff = []
                    poms = set(target['spec']) | set(r['spec'])
                    for p in poms:
                        v1, v2 = target['spec'].get(p, 0), r['spec'].get(p, 0)
                        diff.append({"POM": p, "Mẫu Test": v1, "Mẫu Kho": v2, "Chênh lệch": round(v1 - v2, 2)})
                    
                    df_res = pd.DataFrame(diff)
                    st.dataframe(df_res, use_container_width=True)

                    # Nút xuất Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_res.to_excel(writer, index=False)
                    st.download_button(label="📥 Tải file so sánh Excel", data=output.getvalue(), file_name=f"SoSanh_{r['name']}.xlsx")
        else:
            st.warning(f"⚠️ Không có dữ liệu mẫu **{target['cat']}** nào trong kho để so sánh.")

