# ==========================================================
# AI FASHION PRO V8.2 - GITHUB STORAGE & SUPABASE VECTOR
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= 1. CẤU HÌNH HỆ THỐNG (THAY TẠI ĐÂY) =================
# Supabase (Lấy từ Settings > API)
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

# GitHub (Lấy từ Settings > Developer settings > Tokens classic)
GH_TOKEN = "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" 
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

supabase: Client = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.2", page_icon="👔")

# ================= 2. AI MODEL & GITHUB API =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

def upload_to_github(img_bytes, filename):
    """Tự động đẩy ảnh lên GitHub imgs/ và lấy Link trực tiếp"""
    # Làm sạch tên file để không bị lỗi URL
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
    url = f"https://github.com{GH_REPO}/contents/imgs/{clean_name}.png"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    content = base64.b64encode(img_bytes).decode()
    
    # Kiểm tra nếu file đã tồn tại để lấy mã SHA (cần thiết để ghi đè)
    check = requests.get(url, headers=headers)
    data = {"message": f"Upload {filename}", "content": content, "branch": GH_BRANCH}
    if check.status_code == 200:
        data["sha"] = check.json()["sha"]
        
    res = requests.put(url, headers=headers, json=data)
    if res.status_code in [200, 201]:
        # Link raw để hiển thị trực tiếp trên web
        return f"https://githubusercontent.com{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.png"
    return None

# ================= 3. TRÍCH XUẤT DỮ LIỆU PDF =================
def parse_val(t):
    try:
        f = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not f: return 0
        v = f[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_data(pdf_path):
    try:
        specs, all_text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                all_text += (p.extract_text() or "") + " "
                for tbl in p.extract_tables():
                    for r in tbl:
                        if not r or len(r) < 2: continue
                        line = " | ".join([str(x) for x in r if x]).upper()
                        # Chỉ lấy các dòng chứa từ khóa kỹ thuật
                        if any(k in line for k in ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE']):
                            vals = [parse_val(x) for x in r if parse_val(x) > 0]
                            if vals: specs[line[:80]] = round(float(np.median(vals)), 2)
        
        # ĐIỀU KIỆN 1: ÍT NHẤT 5 DÒNG THÔNG SỐ
        if len(specs) < 5: return {"err": "Thiếu thông số kỹ thuật (POM < 5)"}

        # ĐIỀU KIỆN 2: PHẢI CÓ HÌNH ẢNH
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        if len(img_bytes) < 1000: return {"err": "Không tìm thấy hình ảnh hợp lệ"}

        # PHÂN LOẠI CẤU TRÚC
        cat = "ÁO"
        inseam = next((v for k, v in specs.items() if 'INSEAM' in k), 0)
        if inseam > 0:
            cat = "QUẦN SHORT" if inseam <= 12 else "QUẦN DÀI"
        elif 'DRESS' in all_text.upper() or 'ĐẦM' in all_text.upper():
            cat = "ĐẦM"
        
        return {"spec": specs, "img": img_bytes, "cat": cat}
    except Exception as e:
        return {"err": str(e)}

# ================= 4. GIAO DIỆN SIDEBAR (NẠP KHO) =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    
    # Hiển thị tổng số lượng thực tế trong kho
    try:
        count_res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{count_res.count if count_res.count else 0} mẫu")
    except: st.metric("Tổng mẫu trong kho", "Kết nối lỗi...")

    st.divider()

    if "up_key" not in st.session_state: st.session_state.up_key = 0
    files = st.file_uploader("Nạp file PDF (Nhiều file)", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")

    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        st_text = st.empty()
        success = 0
        
        for idx, f in enumerate(files):
            st_text.text(f"Đang nạp: {f.name} ({idx+1}/{len(files)})")
            p_bar.progress((idx + 1) / len(files))
            
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            
            if "err" in d:
                st.warning(f"Bỏ qua {f.name}: {d['err']}")
                continue
            
            # 1. Đẩy ảnh lên GitHub và lấy Link
            url_github = upload_to_github(d['img'], f.name)
            if not url_github:
                st.error(f"Lỗi đẩy ảnh GitHub: {f.name}")
                continue

            # 2. Tạo Vector AI
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                img_pil = Image.open(io.BytesIO(d['img'])).convert('RGB')
                vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()

            # 3. Lưu vào Supabase
            supabase.table("ai_data").upsert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": d['spec'],
                "img_url": url_github,
                "category": d['cat']
            }, on_conflict="file_name").execute()
            
            success += 1
            os.remove("tmp.pdf")

        st.session_state.up_key += 1 # Reset uploader
        st.success(f"✅ Đã nạp thành công {success} mẫu!")
        st.rerun()

# ================= 5. GIAO DIỆN CHÍNH (SO SÁNH) =================
st.title("👔 AI Fashion Pro V8.2")
test_file = st.file_uploader("Tải file PDF đối chứng (Test)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")

    if "err" in target:
        st.error(f"File Test không đạt chuẩn: {target['err']}")
    else:
        st.info(f"Phân loại mẫu: **{target['cat']}**")
        
        # CHỈ SO SÁNH CÙNG LOẠI (Ví dụ: Quần dài chỉ so với Quần dài)
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            results = []
            for i in db.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    results.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_url']})

            # Lấy top 5 mẫu giống nhất
            top_results = sorted(results, key=lambda x: x['sim'], reverse=True)[:5]

            for r in top_results:
                # Hiện % lên đầu tiêu đề
                with st.expander(f"🎯 ĐỘ GIỐNG: {r['sim']:.1f}% | {r['name']}"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img'], caption="Mẫu Test")
                    c2.image(r['img'], caption="Mẫu trong Kho (GitHub Link)")

                    # Bảng so sánh thông số
                    diff_data = []
                    all_poms = set(target['spec']) | set(r['spec'])
                    for p in all_poms:
                        v1 = target['spec'].get(p, 0)
                        v2 = r['spec'].get(p, 0)
                        diff_data.append({"Thông số (POM)": p, "Mẫu Test": v1, "Mẫu Kho": v2, "Chênh lệch": round(v1-v2, 2)})
                    
                    df_res = pd.DataFrame(diff_data)
                    st.dataframe(df_res, use_container_width=True)

                    # Nút xuất Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_res.to_excel(writer, index=False, sheet_name='SoSanh')
                    st.download_button(label="📥 Tải bảng so sánh Excel", data=output.getvalue(), file_name=f"SoSanh_{r['name']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning(f"⚠️ Kho chưa có mẫu nào thuộc loại **{target['cat']}** để so sánh.")

# Xóa file test sau khi chạy xong
if os.path.exists("test.pdf"): os.remove("test.pdf")
