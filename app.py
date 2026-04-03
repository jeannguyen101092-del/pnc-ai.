import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (ĐIỀN THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Lỗi cấu hình Supabase: {e}")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.3", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM TIỆN ÍCH =================
def compress_to_webp(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=70, method=6)
    return buf.getvalue()

def upload_to_storage(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename) + ".webp"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=clean_name, file=img_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"}
        )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(clean_name)
    except: return None

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

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    inseam, length = specs.get('INSEAM', 0), specs.get('LENGTH', 0)
    if 'CARGO' in txt: return "QUẦN CARGO"
    if inseam >= 22 or length >= 30: return "QUẦN DÀI"
    if 0 < inseam <= 15 or 0 < length <= 22: return "QUẦN SHORT"
    return "ÁO"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    for r in tb:
                        if not r: continue
                        txt_r = " | ".join([str(x) for x in r if x]).upper()
                        key_found = None
                        for k in ['INSEAM','WAIST','HIP','THIGH','LENGTH','CHEST','SHOULDER','SLEEVE']:
                            if k in txt_r: 
                                key_found = k
                                break
                        if key_found:
                            vals = [parse_val(x) for x in r if x and parse_val(x) > 0]
                            if vals: specs[key_found] = round(float(np.median(vals)), 2)
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path)), "name": os.path.basename(pdf_path)}
    except: return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        res_count = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{res_count.count} mẫu")
    except: st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_webp = compress_to_webp(d['img'])
                img_url = upload_to_storage(img_webp, f.name)
                if img_url:
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(Image.open(io.BytesIO(img_webp))).unsqueeze(0)).flatten().numpy().tolist()
                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
        st.success("🏁 Nạp kho thành công!")
        st.rerun()

# ================= GIAO DIỆN CHÍNH =================
st.title("👔 AI Fashion Pro V11.3")

test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
            
            matches = []
            for item in db.data:
                if item.get('vector'):
                    sim = float(cosine_similarity(v_test.reshape(1,-1), np.array(item['vector']).reshape(1,-1))) * 100
                    matches.append({"name": item['file_name'], "sim": sim, "url": item['img_url'], "spec": item['spec_json']})
            
            top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            # --- PHẦN XUẤT EXCEL SO SÁNH (Đã dời về đây) ---
            st.write("### 🔍 Kết quả so sánh:")
            
            # Chuẩn bị dữ liệu Excel cho tất cả các mẫu so sánh
            export_list = []
            for m in top_matches:
                all_keys = set(target['spec'].keys()).union(set(m['spec'].keys()))
                for k in sorted(all_keys):
                    val_test = target['spec'].get(k, 0)
                    val_db = m['spec'].get(k, 0)
                    diff = round(val_test - val_db, 2)
                    export_list.append({
                        "Mẫu đối chiếu": m['name'],
                        "Độ giống AI (%)": round(m['sim'], 1),
                        "Thông số": k,
                        "Giá trị Test": val_test,
                        "Giá trị Kho": val_db,
                        "Chênh lệch (Diff)": diff,
                        "Ghi chú": "Khớp" if abs(diff) < 0.25 else "Lệch"
                    })
            
            if export_list:
                df_compare = pd.DataFrame(export_list)
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                    df_compare.to_excel(writer, index=False, sheet_name='Comparison_Report')
                
                # Nút tải Excel đặt ngay trên bảng so sánh
                st.download_button(
                    label="📥 TẢI BÁO CÁO ĐỐI CHIẾU (EXCEL)",
                    data=excel_buf.getvalue(),
                    file_name=f"So_Sanh_{target['name']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # --- HIỂN THỊ GIAO DIỆN BẢNG ---
            for m in top_matches:
                with st.expander(f"Mẫu: {m['name']} (Khớp {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1: st.image(target['img'], caption="File Test")
                    with c2: st.image(m['url'], caption="Mẫu trong kho")
                    with c3:
                        comp_data = []
                        all_keys = set(target['spec'].keys()).union(set(m['spec'].keys()))
                        for k in sorted(all_keys):
                            v_t = target['spec'].get(k, 0)
                            v_d = m['spec'].get(k, 0)
                            d = round(v_t - v_d, 2)
                            comp_data.append({"Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": d})
                        st.table(pd.DataFrame(comp_data))

        else: st.warning("Không có mẫu cùng loại trong kho.")
