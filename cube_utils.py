"""cube_utils.py - everything in the Rubik's cube solver that is NOT the neural network.

Face order and facelet layout follow the Kociemba convention:
    U R F D L B, each face = 9 stickers row by row, as seen in the unfolded net

                 U
               L F R B
                 D
"""
import itertools
import numpy as np
import cv2
from collections import Counter
from PIL import Image, ImageDraw

COLORS = ["white", "yellow", "red", "orange", "blue", "green"]       # class index order of the CNN
ORDER = ["U", "R", "F", "D", "L", "B"]
SOLVED = "".join(L * 9 for L in ORDER)
STD_SCHEME = {"U": "white", "R": "red", "F": "green", "D": "yellow", "L": "orange", "B": "blue"}
RGB = {"white": (240, 240, 240), "yellow": (250, 215, 40), "red": (200, 30, 40),
       "orange": (250, 130, 20), "blue": (30, 80, 200), "green": (30, 160, 70)}

# --------------------------------------------------------------------------- simulator
def _facelets():
    out = []
    for L in ORDER:
        for r in range(3):
            for c in range(3):
                if L == "U":   pos, n = (c - 1, 1, r - 1), (0, 1, 0)
                elif L == "R": pos, n = (1, 1 - r, 1 - c), (1, 0, 0)
                elif L == "F": pos, n = (c - 1, 1 - r, 1), (0, 0, 1)
                elif L == "D": pos, n = (c - 1, -1, 1 - r), (0, -1, 0)
                elif L == "L": pos, n = (-1, 1 - r, c - 1), (-1, 0, 0)
                else:          pos, n = (1 - c, 1 - r, -1), (0, 0, -1)
                out.append((pos, n))
    return out

_FL = _facelets()
_INDEX = {fl: i for i, fl in enumerate(_FL)}
_AXIS = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}

def _rot_cw(v, a):
    """Rotate vector v by +90 degrees about the outward axis a. With this cube coordinate system
    (x right, y up, z towards the viewer) that is the clockwise quarter turn seen from outside."""
    v, a = np.array(v), np.array(a)
    return tuple(int(x) for x in (np.cross(a, v) + a * np.dot(a, v)))

def _perm(face):
    a = _AXIS[face]
    perm = list(range(54))
    for i, (pos, n) in enumerate(_FL):
        if np.dot(pos, a) == 1:
            perm[_INDEX[(_rot_cw(pos, a), _rot_cw(n, a))]] = i     # sticker i moves to this slot
    return perm

_PERMS = {f: _perm(f) for f in ORDER}

def apply_move(state, move):
    """state: string/list of 54 letters. move: 'R', "R'", 'R2'."""
    face, turns = move[0], {"": 1, "'": 3, "2": 2}[move[1:]]
    s = list(state)
    for _ in range(turns):
        s = _once(s, face)
    return "".join(s)

def _once(s, face):
    new = list(s)
    for src, dst in enumerate(_PERMS[face]):
        new[dst] = s[src]
    return new

def apply_moves(state, moves):
    for m in (moves.split() if isinstance(moves, str) else moves):
        state = apply_move(state, m)
    return state

def random_scramble(rng, n=25):
    moves, last = [], None
    for _ in range(n):
        f = rng.choice([x for x in ORDER if x != last]); last = f
        moves.append(f + rng.choice(["", "'", "2"]))
    return moves

# --------------------------------------------------------------------------- face detection
def _order_corners(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")

def _quad_candidates(gray, min_area):
    """All roughly square 4-corner shapes found in the edge map, as (area, corners)."""
    out = []
    for lo, hi in ((40, 120), (20, 60), (60, 180)):                 # try a few edge thresholds
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) < min_area:
                continue
            polys = []
            for eps in (0.02, 0.03, 0.045, 0.06):
                approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
                if len(approx) == 4:
                    polys.append(approx.reshape(4, 2))
                    break
            fine = cv2.approxPolyDP(hull, 0.012 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if 4 < len(fine) <= 9:                                    # a corner got cut: keep the 4 corners
                best4 = max(itertools.combinations(range(len(fine)), 4),
                            key=lambda idx: cv2.contourArea(fine[list(idx)].astype("float32")))
                polys.append(fine[list(best4)])                       # that enclose the largest area
            for p in polys:
                q = _order_corners(p)
                w1, w2 = np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[0] - q[3])
                h1, h2 = np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[1] - q[2])
                if 0.6 < (w1 + h1) / max(w2 + h2, 1) < 1.6:
                    out.append((cv2.contourArea(q), q))
    return out

def detect_quad(bgr):
    """Cube face corners (top-left, top-right, bottom-right, bottom-left) in the image's pixel coordinates,
    and whether an outline was really found. If not, the centre square of the photo is returned.
    A single sticker is only about 1/9 of the face, so a shape must cover 15% of the photo to count as the face.
    (A picture that is already a tight crop of the face has no outline, so it also gets the centre square,
    which is the right answer for it.)"""
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cands = []
    for area, q in _quad_candidates(gray, 0.15 * h * w):
        touches = (q[:, 0].min() < 4 or q[:, 1].min() < 4 or q[:, 0].max() > w - 5 or q[:, 1].max() > h - 5)
        if area < 0.9 * h * w and not (touches and area > 0.6 * h * w):   # ignore "the whole photo"
            cands.append((area, q))
    if not cands:
        m = min(h, w); y0, x0 = (h - m) / 2, (w - m) / 2
        return np.float32([[x0, y0], [x0 + m, y0], [x0 + m, y0 + m], [x0, y0 + m]]), False
    return max(cands, key=lambda t: t[0])[1].astype("float32"), True

def warp_quad(bgr, quad, size=96):
    """Flatten the 4-corner region `quad` (original pixel coordinates) into a size x size square."""
    pad = max(10, int(0.05 * max(bgr.shape[:2])))
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(np.float32(quad) + pad, dst)
    return cv2.warpPerspective(padded, M, (size, size))

def extract_face(bgr, size=96):
    """Find the cube face and flatten it. Returns (BGR image, found)."""
    quad, found = detect_quad(bgr)
    return warp_quad(bgr, quad, size), found

def sticker_lab(face_bgr):
    """Median Lab colour of the centre third of each of the 9 stickers -> (9, 3) float array."""
    n = face_bgr.shape[0] // 3
    m = max(1, n // 3)
    out = []
    for r in range(3):
        for c in range(3):
            patch = face_bgr[r * n + m:(r + 1) * n - m, c * n + m:(c + 1) * n - m].reshape(-1, 3)
            med = np.median(patch, axis=0).astype(np.uint8).reshape(1, 1, 3)
            out.append(cv2.cvtColor(med, cv2.COLOR_BGR2LAB)[0, 0].astype(float))
    return np.array(out)

def sticker_hsv(face_bgr):
    n = face_bgr.shape[0] // 3
    m = max(1, n // 3)
    out = []
    for r in range(3):
        for c in range(3):
            patch = face_bgr[r * n + m:(r + 1) * n - m, c * n + m:(c + 1) * n - m].reshape(-1, 3)
            med = np.median(patch, axis=0).astype(np.uint8).reshape(1, 1, 3)
            out.append(cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[0, 0].astype(float))
    return np.array(out)

# --------------------------------------------------------------------------- colour decoding
def constrained_decode(probs):
    """probs: (54, 6) CNN probabilities for the stickers in U,R,F,D,L,B order.
    A real cube has exactly 9 stickers of each colour and 6 different centres, so pick the most
    likely assignment that respects that (linear assignment). Returns 54 colour indices."""
    from scipy.optimize import linear_sum_assignment
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    centres = [4 + 9 * k for k in range(6)]
    others = [i for i in range(54) if i not in centres]
    out = np.zeros(54, dtype=int)
    r, c = linear_sum_assignment(-logp[centres])                 # 6 centres -> 6 different colours
    out[np.array(centres)[r]] = c
    cost = -np.repeat(logp[others], 8, axis=1)                   # 48 stickers -> 6 colours x 8 slots
    r, c = linear_sum_assignment(cost)
    out[np.array(others)[r]] = c // 8
    return out

def centre_nearest(lab6):
    """Classical baseline, no training: lab6 is (6 faces, 9, 3). Every sticker gets the colour of the
    nearest face centre. Centres are named by matching them to the standard palette."""
    from scipy.optimize import linear_sum_assignment
    w = np.array([0.5, 1.0, 1.0])
    centres = lab6[:, 4]
    flat = lab6.reshape(54, 3)
    d = (((flat[:, None, :] - centres[None]) * w) ** 2).sum(-1)
    face_of = d.argmin(1)
    std = np.array([cv2.cvtColor(np.uint8([[RGB[c][::-1]]]), cv2.COLOR_BGR2LAB)[0, 0] for c in COLORS], float)
    dc = (((centres[:, None, :] - std[None]) * w) ** 2).sum(-1)
    r, c = linear_sum_assignment(dc)
    name_of_face = dict(zip(r, c))
    return np.array([name_of_face[f] for f in face_of])

# --------------------------------------------------------------------------- state and solver
def build_state(face_colors):
    """face_colors: dict letter -> list of 9 colour names (row by row). Returns the 54-letter string."""
    centre_to_letter = {face_colors[L][4]: L for L in ORDER}
    if len(centre_to_letter) != 6:
        raise ValueError("The 6 centre stickers must be 6 different colours, but they are "
                         + ", ".join(face_colors[L][4] for L in ORDER) + ".")
    counts = Counter(c for L in ORDER for c in face_colors[L])
    if any(counts[c] != 9 for c in COLORS):
        many = [f"{c} ({counts[c]})" for c in COLORS if counts[c] > 9]
        few = [f"{c} ({counts[c]})" for c in COLORS if counts[c] < 9]
        raise ValueError("Each colour must appear exactly 9 times. Too many: " + (", ".join(many) or "none")
                         + ". Too few: " + (", ".join(few) or "none") + ". Fix the wrong stickers in step 3.")
    return "".join(centre_to_letter[c] for L in ORDER for c in face_colors[L])

def _convert_twophase(out):
    """RubikTwoPhase returns e.g. 'R1 U2 F3 (3f)'. Convert to standard notation: R U2 F'."""
    moves = out.split("(")[0].split()
    return " ".join(m[0] + {"1": "", "2": "2", "3": "'"}[m[1]] for m in moves)

def _solve_backend(state):
    """Use the `kociemba` package if it is installed, otherwise the pure-Python `RubikTwoPhase` package."""
    try:
        import kociemba
    except ImportError:
        kociemba = None
    if kociemba is not None:
        return kociemba.solve(state)                       # raises ValueError for an impossible cube
    try:
        import twophase.solver as sv
    except ImportError:
        raise ImportError("No cube solver is installed. Install one of:  pip install kociemba  "
                          "(needs a C compiler on Windows)  or  pip install RubikTwoPhase  (pure Python, "
                          "but its first run builds tables for about 30 minutes).")
    out = sv.solve(state, 20, 5)
    if not isinstance(out, str):                           # the package returns an error code for bad cubes
        raise ValueError("invalid cube")
    return _convert_twophase(out)

def solve(state):
    if state == SOLVED:
        return ""
    try:
        return _solve_backend(state)
    except ValueError:
        raise ValueError("This cube is impossible, so a sticker is probably misread. "
                         "Fix the wrong sticker and try again.")

# --------------------------------------------------------------------------- drawing
_NET_POS = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}

def draw_net(colors54, cell=26, gap=6):
    """colors54: 54 colour names in U,R,F,D,L,B order. Returns a PIL image of the unfolded cube."""
    W, H = 4 * (3 * cell + gap), 3 * (3 * cell + gap)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for k, L in enumerate(ORDER):
        bx, by = _NET_POS[L]
        for i in range(9):
            r, c = divmod(i, 3)
            x = bx * (3 * cell + gap) + c * cell
            y = by * (3 * cell + gap) + r * cell
            d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=RGB[colors54[k * 9 + i]], outline=(40, 40, 40))
    return img

def state_to_colors(state, letter_color):
    return [letter_color[ch] for ch in state]