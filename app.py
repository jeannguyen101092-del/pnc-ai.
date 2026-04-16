import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state: st.session_state['reset_key'] = 0
if 'selected_repo' not in st.session_state: st.session_state['selected_repo'] = None

# ================= 2. AI CORE ENGINE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes): return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds|tol|grade)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. PDF EXTRACTION =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes, summary = {}, None, ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_bytes = pix.tobytes("webp")
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for p in pdf.pages:
                full_txt += (p.extract_text() or "") + "\n"
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    d_col = df.apply(lambda x: x.astype(str).str.len().mean()).idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == d_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, col_idx].head(10)]) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            temp_data = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, col_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 2}
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes, "summary": "Extracted insights from PDF."}
    except: return None

# ================= 4. RENDER COMPARISON =================
def render_comparison(target_data, repo_data):
    st.divider()
    st.subheader("📊 AUDIT RESULTS: ROUND A VS ROUND B")
    
    # 1. Image Comparison
    c1, c2 = st.columns(2)
    with c1: st.image(repo_data['image_url'], caption="ROUND A (REPO)", use_container_width=True)
    with c2: st.image(target_data['img'], caption="ROUND B (NEW UPLOAD)", use_container_width=True)

    # 2. Spec Comparison
    t_specs = target_data.get('all_specs', {})
    r_specs = repo_data.get('spec_json', {})
    
    # Tìm bảng có tên giống nhau hoặc lấy bảng đầu tiên nếu không khớp tên
    common_tables = list(set(t_specs.keys()).intersection(set(r_specs.keys())))
    
    if not common_tables:
        st.warning("⚠️ Table names don't match. Attempting to compare primary data...")
        t_key = list(t_specs.keys())[0] if t_specs else None
        r_key = list(r_specs.keys())[0] if r_specs else None
    else:
        t_key = r_key = st.selectbox("Select Spec Table:", common_tables)

    if t_key and r_key:
        t_d, r_d = t_specs[t_key], r_specs[r_key]
        pts = sorted(list(set(t_d.keys()).intersection(set(r_d.keys()))))
        
        if pts:
            comp_data = []
            for p in pts:
                v_new, v_repo = t_d[p], r_d[p]
                diff = round(v_new - v_repo, 3)
                status = "✅ MATCH" if abs(diff) <= 0.125 else ("❌ ALERT" if abs(diff) >= 0.5 else "⚠️ MINOR")
                comp_data.append({"Point of Measure": p, "New (B)": v_new, "Repo (A)": v_repo, "Diff": diff, "Status": status})
            
            df = pd.DataFrame(comp_data)
            
            # Export Excel Button
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Export to Excel", output.getvalue(), "Audit_Report.xlsx")
            
            st.dataframe(df.style.applymap(lambda v: 'background-color: #ffcccc' if v=="❌ ALERT" else ('background-color: #fff3cd' if v=="⚠️ MINOR" else ''), subset=['Status']), use_container_width=True)
        else:
            st.error("❌ No matching 'Points of Measure' found between files.")
    else:
        st.error("❌ Could not find any specification tables in one of the files.")

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["Audit vs Repository", "Version Control (Compare 2 Rounds)"], horizontal=True)

if mode == "Audit vs Repository":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_pdf_full_logic(f_audit.read())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            df_db = pd.DataFrame(res.data)
            df_db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in df_db['vector']])).flatten()
            top = df_db.sort_values('sim', ascending=False).head(3)
            
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"SELECT {i+1}", key=f"s_{idx}"): st.session_state['selected_repo'] = row.to_dict()
            
            if st.session_state['selected_repo']:
                render_comparison(target, st.session_state['selected_repo'])

else:
    st.subheader("🔄 Version Control: Repo vs New File")
    res = supabase.table("ai_data").select("file_name", "image_url", "spec_json").execute()
    repo_dict = {item['file_name']: item for item in res.data}
    
    col_a, col_b = st.columns(2)
    with col_a:
        sel_name = st.selectbox("Round A (From Repo):", ["-- Select --"] + list(repo_dict.keys()))
    with col_b:
        f_new = st.file_uploader("Round B (Upload New):", type="pdf")

    if sel_name != "-- Select --" and f_new:
        # Quan trọng: Dùng cache hoặc xử lý trực tiếp để đảm bảo target_data không bị mất
        file_bytes = f_new.read()
        target_data = extract_pdf_full_logic(file_bytes)
        if target_data:
            render_comparison(target_data, repo_dict[sel_name])
