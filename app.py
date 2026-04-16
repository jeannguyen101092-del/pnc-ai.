import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

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
        v = match 
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

def extract_insights(text):
    mapping = {"WASH": "Wash Instruction", "FABRIC": "Fabric Spec", "STITCH": "Stitching Detail", "LABEL": "Labeling", "COLOR": "Colorway", "POCKET": "Pocket Detail", "WAIST": "Waistband"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

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
            summary = extract_insights(full_txt)
        return {"all_specs": all_specs, "img": img_bytes, "summary": summary}
    except: return None

# ================= 4. UTILS =================
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit_Report')
    return output.getvalue()

def render_comparison(target_data, repo_data):
    st.divider()
    st.subheader("🖼️ PHASE 1: VISUAL COMPARISON")
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        st.image(target_data['img'], caption="NEW VERSION (ROUND B - Uploaded)", use_container_width=True)
        st.info(f"**Target Insights:**\n\n{target_data['summary']}")
    with v_c2:
        st.image(repo_data['image_url'], caption=f"REPO VERSION (ROUND A - From Repo): {repo_data['file_name']}", use_container_width=True)
        st.success(f"**Repo Insights:**\n\n{repo_data.get('summary_vi', 'No notes available.')}")

    st.subheader("📊 PHASE 2: SPECIFICATION AUDIT")
    t_specs = target_data['all_specs']
    r_specs = repo_data.get('spec_json', repo_data.get('all_specs'))
    
    common_tables = list(set(t_specs.keys()).intersection(set(r_specs.keys())))
    if common_tables:
        sel_tb = st.selectbox("Select Table to Audit:", common_tables)
        t_d, r_d = t_specs[sel_tb], r_specs[sel_tb]
        pts = sorted(list(set(t_d.keys()).intersection(set(r_d.keys()))))
        
        comp_rows = []
        for p in pts:
            diff = round(t_d[p] - r_d[p], 3)
            status = "✅ MATCH" if abs(diff) <= 0.125 else ("❌ ALERT" if abs(diff) >= 0.5 else "⚠️ MINOR")
            comp_rows.append({"Point of Measure": p, "New (B)": t_d[p], "Repo (A)": r_d[p], "Diff": diff, "Status": status})
        
        df_comp = pd.DataFrame(comp_rows)
        st.download_button(label="📥 Export to Excel", data=to_excel(df_comp), file_name=f"Audit_{repo_data['file_name']}.xlsx", mime="application/vnd.ms-excel")
        st.dataframe(df_comp.style.applymap(lambda v: 'background-color: #ffcccc' if v=="❌ ALERT" else ('background-color: #fff3cd' if v=="⚠️ MINOR" else ''), subset=['Status']), use_container_width=True)
    else:
        st.warning("⚠️ No matching table names found for automated audit.")

# ================= 5. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Models", f"{count}")
    
    used_mb = count * 0.08
    st.write(f"💾 **Storage:** {used_mb:.1f}MB / 1024MB")
    st.progress(min((used_mb / 1024), 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNC TO REPO", use_container_width=True):
        for f in new_files:
            fb = f.read(); h = get_file_hash(fb)
            data = extract_pdf_full_logic(fb)
            if data:
                path = f"lib_{h}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "summary_vi": data['summary'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.rerun()
    if st.button("CLEAR LIST", use_container_width=True): st.session_state['reset_key'] += 1; st.rerun()

# ================= 6. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["Audit vs Repository", "Version Control (Compare 2 Rounds)"], horizontal=True)

if mode == "Audit vs Repository":
    f_audit = st.file_uploader("Upload Target PDF to compare with AI Repo:", type="pdf")
    if f_audit:
        target = extract_pdf_full_logic(f_audit.read())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            db = pd.DataFrame(res.data)
            db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in db['vector']])).flatten()
            top = db.sort_values('sim', ascending=False).head(3)
            
            st.subheader("🎯 AI Search Results")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], caption=f"Match Score: {row['sim']:.1%}")
                    if st.button(f"SELECT MODEL {i+1}", key=f"s_{idx}"): st.session_state['selected_repo'] = row.to_dict()
            
            if st.session_state['selected_repo']:
                render_comparison(target, st.session_state['selected_repo'])

# --- PHẦN BẠN YÊU CẦU: SELECT REPO VS UPLOAD NEW FILE ---
else:
    st.subheader("🔄 Version Control: Compare 2 Style Rounds")
    # Lấy danh sách từ kho (Repo)
    res = supabase.table("ai_data").select("file_name", "image_url", "spec_json", "summary_vi").execute()
    repo_dict = {item['file_name']: item for item in res.data}
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("🔍 **Select Version A (Search in Repo):**")
        sel_name = st.selectbox("Choose a style from repository:", ["-- Choose from Repo --"] + list(repo_dict.keys()), label_visibility="collapsed")
        if sel_name != "-- Choose from Repo --":
            st.image(repo_dict[sel_name]['image_url'], caption="Round A (Repo Reference)", use_container_width=True)

    with col_right:
        st.write("📤 **Select Version B (Upload New File):**")
        f_new = st.file_uploader("Upload round B PDF to compare:", type="pdf", label_visibility="collapsed")
        if f_new:
            # Hiển thị preview ngay sau khi up
            with st.spinner("Processing New File..."):
                target_data = extract_pdf_full_logic(f_new.read())
                if target_data:
                    st.image(target_data['img'], caption="Round B (New Upload)", use_container_width=True)

    # Nếu cả 2 bên đã sẵn sàng, hiển thị bảng so sánh chi tiết
    if sel_name != "-- Choose from Repo --" and f_new:
        if 'target_data' in locals():
            render_comparison(target_data, repo_dict[sel_name])
