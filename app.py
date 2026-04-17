import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Hybrid Auditor", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. DNA LOGIC (NEW) =================

def extract_dna_from_text(text):
    """Chuyển đổi văn bản thô thành cấu trúc JSON DNA chuẩn"""
    text = text.upper()
    dna = {
        "pockets": {"front": "unknown", "back": "unknown", "extra": "none"},
        "construction": {"closure": "zipper", "belt_loop": "yes"},
        "fit": {"shape": "regular", "leg": "straight"},
        "material": {"fabric": "woven"},
        "appearance": {"color": "unknown"}
    }
    # Logic Pocket (40%)
    if any(x in text for x in ["SLANT", "SIDE POCKET"]): dna["pockets"]["front"] = "slanted"
    elif any(x in text for x in ["CURVE", "SCOOP", "JEAN POCKET"]): dna["pockets"]["front"] = "curved"
    elif "FROG" in text: dna["pockets"]["front"] = "frog"
    if "PATCH" in text and "BACK" in text: dna["pockets"]["back"] = "patch"
    elif "WELT" in text and "BACK" in text: dna["pockets"]["back"] = "welt"
    if "COIN" in text: dna["pockets"]["extra"] = "coin"
    if "CARGO" in text: dna["pockets"]["extra"] = "cargo"
    # Construction (20%)
    if "BUTTON" in text and "FLY" in text: dna["construction"]["closure"] = "button"
    return dna

def calculate_hybrid_sim(t_dna, t_vec, r_data):
    """Hàm so sánh lai: Ưu tiên DNA, nếu không có thì dùng Vector"""
    r_dna = r_data.get("dna_json")
    r_vec = r_data.get("vector")
    
    # Nếu cả 2 đều có DNA -> Dùng logic 40/20/15/15/10
    if t_dna and r_dna:
        score = 0.0
        p_score = 0
        if t_dna["pockets"]["front"] == r_dna["pockets"]["front"]: p_score += 0.4
        if t_dna["pockets"]["back"] == r_dna["pockets"]["back"]: p_score += 0.4
        if t_dna["pockets"]["extra"] == r_dna["pockets"]["extra"]: p_score += 0.2
        score += (p_score * 0.4)
        if t_dna["construction"] == r_dna["construction"]: score += 0.2
        if t_dna["fit"] == r_dna["fit"]: score += 0.15
        if t_dna["material"] == r_dna["material"]: score += 0.15
        if t_dna["appearance"] == r_dna["appearance"]: score += 0.10
        return score
    
    # Nếu không có DNA -> Dùng Vector hình ảnh (Dữ liệu cũ 1000 file)
    if t_vec is not None and r_vec is not None:
        return float(cosine_similarity(np.array(t_vec).reshape(1,-1), np.array(r_vec).reshape(1,-1))[0][0])
    
    return 0.0

# ================= 3. AI CORE & SCRAPER (GIỮ NGUYÊN PHẦN THÔNG SỐ CỦA BẠN) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page", "tol"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        return float(num[0]) if num else 0
    except: return 0

def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes, all_text = {}, None, ""
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_pil.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue(); doc.close()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                all_text += txt + " "
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text'])
                    line_vals = [parse_val(w) for w in sorted_group['text'] if parse_val(w) > 0]
                    if any(kw in line_txt.upper() for kw in POM_KWS) and line_vals:
                        pom_name = re.sub(r'[\d./\s]+$', '', line_txt).strip()
                        for i, val in enumerate(line_vals):
                            s_key = f"Size_{i+1}"
                            if s_key not in all_specs: all_specs[s_key] = {}
                            all_specs[s_key][pom_name] = val
        
        return {"all_specs": all_specs, "img": img_bytes, "dna": extract_dna_from_text(all_text), "vector": get_vector(img_bytes)}
    except: return None

# ================= 4. UI & SYNC =================
with st.sidebar:
    st.header("PPJ Hybrid System")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Models in Repo", f"{res_count.count or 0} SKUs")
    
    new_files = st.file_uploader("Upload New Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNC NEW DATA"):
        for f in new_files:
            data = extract_full_data(f.getvalue())
            if data:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": data['vector'],
                    "spec_json": data['all_specs'], "dna_json": data['dna'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR (HYBRID DNA)")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            if res.data:
                valid_matches = []
                for r in res.data:
                    # SO SÁNH LAI: DNA hoặc Vector
                    sim = calculate_hybrid_sim(target['dna'], target['vector'], r)
                    valid_matches.append({**r, "sim_score": sim})
                
                df_db = pd.DataFrame(valid_matches).sort_values('sim_score', ascending=False).head(3)
                
                st.subheader("🎯 Top Matches (Hybrid Logic)")
                cols = st.columns(4)
                cols[0].image(target['img'], caption="TARGET PDF")
                for i, (idx, row) in enumerate(df_db.iterrows()):
                    with cols[i+1]:
                        # Hiển thị loại so khớp
                        match_type = "🧬 DNA Match" if row.get("dna_json") else "🖼️ Visual Match"
                        st.image(row['image_url'], caption=f"{match_type}: {row['sim_score']:.1%}")
                        if st.button(f"SELECT {i+1}", key=f"s_{idx}"):
                            st.session_state['sel_audit'] = row

            if st.session_state['sel_audit']:
                sel = st.session_state['sel_audit']
                st.success(f"📈 Comparing with: **{sel['file_name']}**")
                # Hiển thị thông số Size (Giữ nguyên cấu trúc của bạn)
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = sel.get('spec_json', {}).get(get_close_matches(sz, list(sel.get('spec_json', {}).keys()), 1, 0.4)[0] if get_close_matches(sz, list(sel.get('spec_json', {}).keys()), 1, 0.4) else "", {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows: r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                        st.table(pd.DataFrame(rows))
