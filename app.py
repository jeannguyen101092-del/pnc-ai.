import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG (Thay bằng thông tin của bạn) =================
URL = "https://supabase.co"
KEY = "your-anon-key"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.46", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM CHỤP ẢNH EXCEL SIÊU NÉT (DPI 500) =================
def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).dropna(how='all', axis=0).fillna("")
        df_display = df.head(80)
        fig, ax = plt.subplots(figsize=(24, len(df_display) * 0.6 + 2)) 
        ax.axis('off')
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(16) 
        table.scale(1.2, 3.2) 
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', size=18)
                cell.set_facecolor('#000000')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=500)
        plt.close(fig)
        return buf.getvalue()
    except: return None

# ================= TRÍCH XUẤT THÔNG SỐ (LẤY CỘT CÓ GIÁ TRỊ LỚN NHẤT) =================
def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    if not tb or len(tb) < 2: continue
                    
                    # LOGIC: Tìm cột có giá trị trung bình lớn nhất (Bỏ qua cột 0 và cột Dung sai)
                    col_stats = {}
                    for i in range(2, len(tb[0])):
                        vals = [parse_val(r[i]) for r in tb[1:] if len(r) > i]
                        # Chỉ tính các cột có giá trị nằm trong khoảng kích thước quần thực tế
                        valid_vals = [v for v in vals if 5.0 < v < 60.0]
                        if valid_vals:
                            col_stats[i] = sum(valid_vals) / len(valid_vals)
                    
                    if not col_stats: continue
                    base_idx = max(col_stats, key=col_stats.get)

                    for r in tb[1:]:
                        if not r or len(r) <= base_idx: continue
                        desc = (str(r[1] or "") + " " + str(r[2] or "")).strip().upper().replace("\n", " ")
                        val = parse_val(r[base_idx])
                        if val > 1.0 and len(desc) > 5 and not any(x in desc for x in ['DATE', 'PAGE']):
                            specs[desc[:150]] = round(float(val), 3)
                            
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

def classify_logic(specs, text, name):
    txt = (text + " " + name).upper()
    length = 0
    if specs:
        lv = [v for k,v in specs.items() if 'LENGTH' in k or 'OUTSEAM' in k]
        if lv: length = max(lv)
    if 'SHORT' in txt or (0 < length < 24): return "QUẦN SHORT"
    return "QUẦN DÀI" if any(k in txt for k in ['PANT', 'TROUSER']) or length >= 24 else "ÁO / KHÁC"

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("*").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF & Excel (Cùng mã)", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        groups = {}
        for f in files:
            m = re.search(r'^\d+', f.name)
            if m:
                ma = m.group(); ext = os.path.splitext(f.name)[1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f
        
        for ma, parts in groups.items():
            f_p, f_e = parts.get('.pdf'), (parts.get('.xlsx') or parts.get('.xls'))
            if f_p and f_e:
                with st.spinner(f"Nạp mã: {ma}"):
                    with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                    d = get_data("tmp.pdf")
                    exl = excel_to_img_bytes(f_e)
                    if d and exl:
                        img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                        buf = io.BytesIO(); img_p.save(buf, format="WEBP")
                        supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_t.webp", buf.getvalue(), {"upsert":"true"})
                        url_t = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_t.webp")
                        supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_e.webp", exl, {"upsert":"true"})
                        url_e = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_e.webp")
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                        with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                        supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": url_t, "excel_img_url": url_e, "category": d['cat']}, on_conflict="file_name").execute()
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.rerun()

# ================= CHÍNH =================
st.title("👔 AI Fashion Pro V11.46")
test_file = st.file_uploader("Tải PDF Test", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện: **{target['cat']}**")
        same_cat = [i for i in all_samples if i['category'] == target['cat']]
        if same_cat:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_t = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().detach().numpy().reshape(1, -1)
            matches = []
            for item in same_cat:
                if item.get('vector'):
                    v_raw = item['vector']
                    if isinstance(v_raw, str): v_raw = [float(x) for x in v_raw.strip('[]').split(',')]
                    v_db = np.array(v_raw, dtype=np.float32).reshape(1, -1)
                    # SỬA LỖI: Lấy giá trị float từ matrix
                    sim = float(cosine_similarity(v_t, v_db)[0][0]) * 100
                    matches.append(item | {"sim": sim})
            
            for m in sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['file_name']} ({m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Test", use_container_width=True)
                    with c2: 
                        st.image(m['img_url'], caption="Kho", use_container_width=True)
                        if m.get('excel_img_url'): st.image(m['excel_img_url'], caption="Định mức", use_container_width=True)
                    with c3:
                        res = []
                        if target['spec'] and m['spec_json']:
                            for kt, vt in target['spec'].items():
                                mk = next((k for k in m['spec_json'].keys() if SequenceMatcher(None, kt, k).ratio() > 0.8), None)
                                vd = m['spec_json'][mk] if mk else 0.0
                                res.append({"Thông số": kt, "Test": vt, "Kho": vd, "Lệch": round(vt - vd, 3)})
                            df_final = pd.DataFrame(res)
                            st.table(df_final)
                            out = io.BytesIO()
                            with pd.ExcelWriter(out, engine='xlsxwriter') as wr: df_final.to_excel(wr, index=False)
                            st.download_button(f"📥 XUẤT EXCEL: {m['file_name']}", out.getvalue(), f"SoSanh_{m['file_name']}.xlsx")

if os.path.exists("test.pdf"): os.remove("test.pdf")
gc.collect()
