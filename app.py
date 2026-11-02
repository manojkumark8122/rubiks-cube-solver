"""Rubik's Cube Solver - Streamlit app.
Pipeline: photo -> find the face -> CNN reads the 9 sticker colours -> 54-sticker state -> Kociemba solver."""
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from cube_utils import (COLORS, ORDER, SOLVED, STD_SCHEME, RGB, detect_quad, warp_quad, sticker_lab, constrained_decode,
                        centre_nearest, build_state, solve, apply_moves, draw_net, state_to_colors)
 
st.set_page_config(page_title="Rubik's Cube Solver", page_icon="🧩", layout="wide")
MODEL_PATH = "face_cnn.keras"
DEMO = "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD"     # a scrambled cube for trying the app
 
GUIDE = {
    "F": "Hold the cube so ONE face looks at the camera. This face is F. Another face must be on TOP (this will be U).",
    "R": "Turn the whole cube so the face that was on your RIGHT now looks at the camera. Keep U on top.",
    "B": "Turn the cube the same way again (next face on the right comes to the camera). Keep U on top.",
    "L": "Turn the cube the same way once more. Keep U on top.",
    "U": "Go back to the F position, then tilt the cube so the TOP face looks at the camera.",
    "D": "Go back to the F position, then tilt the cube so the BOTTOM face looks at the camera.",
}
 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH, compile=False)
 
model = load_model()
st.session_state.setdefault("raw", {})            # face letter -> jpeg/png bytes
st.session_state.setdefault("lowconf", [])

def show_html(html, height):
    """Works on old and new Streamlit: st.iframe is the new way, components.html the old one."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)

def cube3d_html(colors54, size=220):
    """A cube you can drag with the mouse. colors54 = 54 colour names in U,R,F,D,L,B order."""
    half = size // 2
    place = {"F": f"translateZ({half}px)", "B": f"rotateY(180deg) translateZ({half}px)",
             "R": f"rotateY(90deg) translateZ({half}px)", "L": f"rotateY(-90deg) translateZ({half}px)",
             "U": f"rotateX(90deg) translateZ({half}px)", "D": f"rotateX(-90deg) translateZ({half}px)"}
    faces = ""
    for k, L in enumerate(ORDER):
        cells = "".join(f'<div style="background:rgb{RGB[colors54[k * 9 + i]]}"></div>' for i in range(9))
        faces += f'<div class="f" style="transform:{place[L]}">{cells}</div>'
    page = """
    <style>
      .f{position:absolute;width:__S__px;height:__S__px;display:grid;grid-template-columns:repeat(3,1fr);
         gap:4px;padding:4px;box-sizing:border-box;background:#111;border-radius:8px}
      .f div{border-radius:6px}
    </style>
    <div id="scene" style="height:__H__px;display:flex;align-items:center;justify-content:center;
         perspective:900px;cursor:grab;user-select:none;touch-action:none">
      <div id="cube" style="position:relative;width:__S__px;height:__S__px;transform-style:preserve-3d;
           transform:rotateX(-28deg) rotateY(-38deg)">__FACES__</div>
    </div>
    <script>
      const cube = document.getElementById('cube'), scene = document.getElementById('scene');
      let rx = -28, ry = -38, drag = false, px = 0, py = 0;
      const apply = () => cube.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg)`;
      scene.addEventListener('pointerdown', e => { drag = true; px = e.clientX; py = e.clientY; scene.setPointerCapture(e.pointerId); });
      scene.addEventListener('pointermove', e => { if (!drag) return; ry += (e.clientX - px) * 0.6; rx -= (e.clientY - py) * 0.6;
                                                    px = e.clientX; py = e.clientY; apply(); });
      scene.addEventListener('pointerup', () => drag = false);
    </script>"""
    return page.replace("__S__", str(size)).replace("__H__", str(size + 90)).replace("__FACES__", faces)
 
# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    reader = "CNN (plain)"
    if model is None:
        st.error("face_cnn.keras not found. Put it next to app.py.")
    mode = st.radio("Input", ["Upload photos", "Webcam"])
    st.caption("Model loaded: " + ("yes" if model is not None else "no (put face_cnn.keras next to app.py)"))
    if st.button("Load demo cube"):
        for k, L in enumerate(ORDER):
            for i in range(9):
                st.session_state[f"edit_{L}_{i}"] = STD_SCHEME[DEMO[k * 9 + i]]
        st.session_state["raw"] = {}
    if st.button("Reset all"):
        st.session_state.clear(); st.rerun()
    st.markdown("**Tips:** plain background, good light, face filling most of the picture, "
                "hold the cube straight. A wrongly read sticker can always be fixed in step 3.")
 
st.title("🧩 Rubik's Cube Solver")
st.caption("Deep learning reads the colours, the Kociemba algorithm solves the cube.")
 
# ---------------------------------------------------------------- step 1: capture faces
def decode_bytes(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
 
st.subheader("1. Scan the 6 faces")
left, right = st.columns([1, 1])
with left:
    missing = [L for L in ORDER if L not in st.session_state["raw"]]
    face = st.radio("Face to scan", ORDER, index=ORDER.index(missing[0]) if missing else 0, horizontal=True)
    st.info(f"**{face}**: " + GUIDE[face])
    if mode == "Upload photos":
        f = st.file_uploader(f"Photo of face {face}", type=["jpg", "jpeg", "png", "webp"], key=f"up_{face}")
    else:
        f = st.camera_input(f"Take a photo of face {face}", key=f"cam_{face}")
    if f is not None:
        st.session_state["raw"][face] = f.getvalue()
 
# remember the face outline (4 corners) of every photo; auto-detect it when a new photo arrives
quads = st.session_state.setdefault("quads", {})
for L, b in st.session_state["raw"].items():
    sig = (len(b), hash(b[:4096]))
    if L not in quads or quads[L]["sig"] != sig:
        img = decode_bytes(b)
        if img is not None:
            q, found = detect_quad(img)
            quads[L] = dict(sig=sig, quad=q, auto=q.copy(), found=found, ver=0, shape=img.shape[:2])
 
CORNERS = ["top-left", "top-right", "bottom-right", "bottom-left"]
with left:
    if face in quads:
        qd = quads[face]; h, w = qd["shape"]
        with st.expander("Face outline: adjust the 4 corners if the flattened face looks wrong",
                         expanded=not qd["found"]):
            c1, c2 = st.columns(2); new = []
            for i, name in enumerate(CORNERS):
                x0, y0 = qd["quad"][i]
                nx = c1.slider(f"{name} x (%)", 0.0, 100.0, float(np.clip(x0 / w * 100, 0, 100)), 0.5,
                               key=f"qx_{face}_{qd['ver']}_{i}")
                ny = c2.slider(f"{name} y (%)", 0.0, 100.0, float(np.clip(y0 / h * 100, 0, 100)), 0.5,
                               key=f"qy_{face}_{qd['ver']}_{i}")
                new.append([nx / 100 * w, ny / 100 * h])
            qd["quad"] = np.float32(new)
            b1, b2 = st.columns(2)
            if b1.button("Reset to auto-detected", key=f"reset_{face}"):
                qd["quad"] = qd["auto"].copy(); qd["ver"] += 1; st.rerun()
            if b2.button("Use whole photo", key=f"whole_{face}"):
                qd["quad"] = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); qd["ver"] += 1; st.rerun()
            shown = decode_bytes(st.session_state["raw"][face]).copy()
            cv2.polylines(shown, [qd["quad"].astype(int)], True, (0, 255, 0), max(2, w // 200))
            st.image(shown[..., ::-1], caption="Green outline = the area that is flattened and read")
 
faces = {}
for L, b in st.session_state["raw"].items():
    img = decode_bytes(b)
    if img is not None and L in quads:
        qd = quads[L]
        adjusted = not np.allclose(qd["quad"], qd["auto"], atol=1.0)
        faces[L] = (warp_quad(img, qd["quad"], 96), qd["found"] or adjusted)
with right:
    cols = st.columns(6)
    for c, L in zip(cols, ORDER):
        if L in faces:
            c.image(faces[L][0][..., ::-1], caption=f"{L} " + ("✓" if faces[L][1] else "⚠"), width=90)
        else:
            c.markdown(f"**{L}**\n\n_missing_")
    if any(not found for _, found in faces.values()):
        st.warning("⚠ For some photos the face outline was not found automatically (the centre of the photo is "
                   "used). If a flattened face looks wrong, open 'Face outline' on the left and adjust the corners.")
 
# ---------------------------------------------------------------- step 2: read colours
st.subheader("2. Read the colours")
if st.button("Read colours", disabled=len(faces) < 6):
    bgr = np.stack([faces[L][0] for L in ORDER])
    low = []
    if reader.startswith("CNN"):
        probs = model.predict(bgr[..., ::-1].copy(), verbose=0).reshape(54, 6)
        idx = constrained_decode(probs) if "constrained" in reader else probs.argmax(1)
        low = [f"{ORDER[i // 9]}{i % 9 + 1}" for i in range(54) if probs[i, idx[i]] < 0.7]
    else:
        idx = centre_nearest(np.stack([sticker_lab(f) for f in bgr]))
    for k, L in enumerate(ORDER):
        for i in range(9):
            st.session_state[f"edit_{L}_{i}"] = COLORS[idx[k * 9 + i]]
    st.session_state["lowconf"] = low
if len(faces) < 6:
    st.caption(f"Scan all 6 faces first ({6 - len(faces)} missing). Or use 'Load demo cube' in the sidebar.")
if st.session_state["lowconf"]:
    st.warning("Low-confidence stickers, please check: " + ", ".join(st.session_state["lowconf"]))
 
# ---------------------------------------------------------------- step 3: review and fix
st.subheader("3. Check and fix the colours")
ready = all(f"edit_{L}_{i}" in st.session_state for L in ORDER for i in range(9))
net_slot = st.container()
if ready:
    with st.expander("Edit stickers (rows as seen when you scanned the face)", expanded=False):
        cols = st.columns(6)
        for c, L in zip(cols, ORDER):
            c.markdown(f"**{L}**")
            for i in range(9):
                c.selectbox(f"{L}{i + 1}", COLORS, key=f"edit_{L}_{i}", label_visibility="collapsed")
    current = {L: [st.session_state[f"edit_{L}_{i}"] for i in range(9)] for L in ORDER}
    colors_now = [c for L in ORDER for c in current[L]]
    with net_slot:
        n1, n2 = st.columns(2)
        n1.image(draw_net(colors_now), caption="Unfolded cube as read")
        with n2:
            show_html(cube3d_html(colors_now), height=320)
    net_slot.caption("The unfolded cube appears here after step 2.")
 
# ---------------------------------------------------------------- step 4: solve
st.subheader("4. Solve")
if st.button("Solve the cube", disabled=not ready, type="primary"):
    try:
        state = build_state(current)
        moves = solve(state)
        if moves and apply_moves(state, moves) != SOLVED:
            raise ValueError("This cube is impossible, so a sticker is probably misread. Fix the wrong sticker and try again.")
        st.session_state["solution"] = dict(state=state, moves=moves.split(),
                                            letter_color={L: current[L][4] for L in ORDER})
    except (ValueError, ImportError) as e:
        st.session_state.pop("solution", None)
        st.error(str(e))
 
sol = st.session_state.get("solution")
if sol:
    if not sol["moves"]:
        st.success("The cube is already solved.")
    else:
        st.success(f"Solved in {len(sol['moves'])} moves")
        st.code(" ".join(sol["moves"]))
        step = st.slider("Step through the solution", 0, len(sol["moves"]), 0)
        now = apply_moves(sol["state"], sol["moves"][:step])
        c1, c2 = st.columns([1, 1])
        now_colors = state_to_colors(now, sol["letter_color"])
        c1.image(draw_net(now_colors), caption=f"After {step} move(s)")
        with c2:
            show_html(cube3d_html(now_colors), height=320)
        c2.markdown("**Next move:** " + (f"`{sol['moves'][step]}`" if step < len(sol["moves"]) else "none, solved!"))
        with st.expander("How to read the moves"):
            st.markdown("Letters are faces: **U**p, **D**own, **L**eft, **R**ight, **F**ront, **B**ack. "
                        "A letter alone = turn that face 90° clockwise (looking at that face). "
                        "`'` = anticlockwise, `2` = half turn. Hold the cube with the same U and F as when you scanned it.")
 