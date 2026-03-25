import streamlit as st
import hashlib
import heapq
from collections import deque
import json
import re

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blind Search Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

OPERATORS_BASE = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

CELL_COLORS = {
    "IS":       "#4CAF50",
    "FS":       "#9C27B0",
    "OBSTACLE": "#F44336",
    "PATH":     "#2196F3",
    "VISITED":  "#FFF9C4",
    "CURRENT":  "#FF9800",
    "DEFAULT":  "#FFFFFF",
    "FRONTIER": "#BBDEFB",
}

ALGO_COLORS = {
    "BFS": "#1565C0",
    "DFS": "#B71C1C",
    "DLS": "#E65100",
    "IDS": "#1B5E20",
    "UCS": "#4A148C",
    "BDS": "#006064",
}

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_params(nim5: str) -> dict:
    """Generate deterministic params from 5-digit NIM."""
    seed = int(hashlib.md5(nim5.encode()).hexdigest(), 16)

    def pick(lst, offset):
        return lst[(seed >> offset) % len(lst)]

    def pick_n(n, lst, offset):
        s = (seed >> offset) % (10**9)
        perm = lst[:]
        result = []
        for i in range(n):
            idx = s % len(perm)
            result.append(perm.pop(idx))
            s //= max(1, len(perm) + 1)
        return result

    all_cells = [(r, c) for r in range(1, 4) for c in range(1, 4)]

    # IS choices: corners and edges, not (2,2)
    is_choices = [(1,1),(1,3),(3,1),(3,3),(1,2),(2,1),(2,3),(3,2)]
    IS = is_choices[(seed >> 2) % len(is_choices)]

    # FS must be far from IS (at least 3 steps Manhattan distance)
    fs_choices = [c for c in is_choices if abs(c[0]-IS[0])+abs(c[1]-IS[1]) >= 3]
    if not fs_choices:
        fs_choices = [c for c in is_choices if c != IS and abs(c[0]-IS[0])+abs(c[1]-IS[1]) >= 2]
    FS = fs_choices[(seed >> 5) % len(fs_choices)]

    # Obstacle: not IS or FS, preferably interior
    obs_choices = [c for c in all_cells if c != IS and c != FS]
    OBSTACLE = obs_choices[(seed >> 8) % len(obs_choices)]

    # Verify path exists (simple check)
    def has_path(start, goal, obs):
        visited = set()
        q = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            if cur == goal:
                return True
            for dr, dc in MOVES.values():
                nb = (cur[0]+dr, cur[1]+dc)
                if 1<=nb[0]<=3 and 1<=nb[1]<=3 and nb not in visited and nb != obs:
                    visited.add(nb)
                    q.append(nb)
        return False

    # Retry obstacle if no path
    for attempt in range(len(obs_choices)):
        OBSTACLE = obs_choices[(seed >> 8 + attempt) % len(obs_choices)]
        if OBSTACLE != IS and OBSTACLE != FS and has_path(IS, FS, OBSTACLE):
            break

    # Operator order
    op_perm = pick_n(4, OPERATORS_BASE[:], 12)

    # Cost grid: each cell gets cost 1, 2, or 3
    cost_grid = {}
    for i, cell in enumerate(all_cells):
        base_cost = 1 + ((seed >> (i * 3)) % 3)
        cost_grid[cell] = base_cost
        
    # Pastikan tidak semua cell normal memiliki cost yang sama
    normal_cells = [c for c in all_cells if c not in (IS, FS, OBSTACLE)]
    unique_costs = set(cost_grid[c] for c in normal_cells)
    if len(unique_costs) == 1:
        val = cost_grid[normal_cells[0]]
        cost_grid[normal_cells[0]] = 1 if val != 1 else 2
        cost_grid[normal_cells[-1]] = 3 if val != 3 else 2

    cost_grid[IS] = 1
    cost_grid[FS] = 1
    cost_grid[OBSTACLE] = 99  # impassable

    return {
        "nim": nim5,
        "IS": IS,
        "FS": FS,
        "OBSTACLE": OBSTACLE,
        "operators": op_perm,
        "cost_grid": cost_grid,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ALGORITHMS (step-by-step)
# ─────────────────────────────────────────────────────────────────────────────
def get_neighbors(node, obstacle, operators):
    neighbors = []
    for op in operators:
        dr, dc = MOVES[op]
        nb = (node[0]+dr, node[1]+dc)
        if 1<=nb[0]<=3 and 1<=nb[1]<=3 and nb != obstacle:
            neighbors.append((op, nb))
    return neighbors

def bfs_steps(IS, FS, obstacle, operators):
    steps = []
    queue = deque([(IS, [IS])])
    visited = {IS}
    while queue:
        node, path = queue.popleft()
        neighbors = get_neighbors(node, obstacle, operators)
        added = []
        for op, nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
                added.append(nb)
        steps.append({
            "expanded": node,
            "queue": list(q[0] for q in queue),
            "visited": set(visited),
            "added": added,
            "path_so_far": path,
            "found": node == FS,
        })
        if node == FS:
            return steps, path
    return steps, None

def dfs_steps(IS, FS, obstacle, operators):
    steps = []
    # Use reversed operators so pop() gives priority order
    stack = [(IS, [IS])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        neighbors = get_neighbors(node, obstacle, operators)
        added = []
        for op, nb in reversed(neighbors):  # reversed so first-priority ends on top
            if nb not in visited:
                stack.append((nb, path + [nb]))
                added.append(nb)
        steps.append({
            "expanded": node,
            "stack": [s[0] for s in stack],
            "visited": set(visited),
            "added": added,
            "path_so_far": path,
            "found": node == FS,
        })
        if node == FS:
            return steps, path
    return steps, None

def dls_steps(IS, FS, obstacle, operators, limit):
    steps = []
    stack = [(IS, [IS], 0)]
    visited = set()
    while stack:
        node, path, depth = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        cutoff = depth >= limit
        added = []
        if not cutoff:
            neighbors = get_neighbors(node, obstacle, operators)
            for op, nb in reversed(neighbors):
                if nb not in visited:
                    stack.append((nb, path + [nb], depth + 1))
                    added.append(nb)
        steps.append({
            "expanded": node,
            "depth": depth,
            "cutoff": cutoff,
            "stack": [s[0] for s in stack],
            "visited": set(visited),
            "added": added,
            "path_so_far": path,
            "found": node == FS,
        })
        if node == FS:
            return steps, path
    return steps, None

def ids_steps(IS, FS, obstacle, operators):
    all_iterations = []
    for limit in range(0, 10):
        steps, path = dls_steps(IS, FS, obstacle, operators, limit)
        all_iterations.append({"limit": limit, "steps": steps, "path": path})
        if path is not None:
            return all_iterations
    return all_iterations

def ucs_steps(IS, FS, obstacle, operators, cost_grid):
    steps = []
    pq = [(0, IS, [IS])]
    visited = {}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited[node] = cost
        neighbors = get_neighbors(node, obstacle, operators)
        added = []
        for op, nb in neighbors:
            if nb not in visited:
                new_cost = cost + cost_grid.get(nb, 1)
                heapq.heappush(pq, (new_cost, nb, path + [nb]))
                added.append((nb, new_cost))
        steps.append({
            "expanded": node,
            "g": cost,
            "pq": [(p[0], p[1]) for p in pq],
            "visited": dict(visited),
            "added": added,
            "path_so_far": path,
            "found": node == FS,
        })
        if node == FS:
            return steps, path, cost
    return steps, None, None

def bds_steps(IS, FS, obstacle, operators):
    steps = []
    fwd_queue = deque([IS])
    bwd_queue = deque([FS])
    fwd_visited = {IS: [IS]}
    bwd_visited = {FS: [FS]}
    meeting = None

    def check_meeting():
        inter = set(fwd_visited.keys()) & set(bwd_visited.keys())
        return list(inter)[0] if inter else None

    while fwd_queue or bwd_queue:
        # Forward step
        if fwd_queue:
            node = fwd_queue.popleft()
            path = fwd_visited[node]
            added_f = []
            for op, nb in get_neighbors(node, obstacle, operators):
                if nb not in fwd_visited:
                    fwd_visited[nb] = path + [nb]
                    fwd_queue.append(nb)
                    added_f.append(nb)
            meeting = check_meeting()
            steps.append({
                "direction": "FORWARD",
                "expanded": node,
                "fwd_queue": list(fwd_queue),
                "bwd_queue": list(bwd_queue),
                "fwd_visited": set(fwd_visited.keys()),
                "bwd_visited": set(bwd_visited.keys()),
                "added": added_f,
                "meeting": meeting,
                "found": meeting is not None,
            })
            if meeting:
                fwd_path = fwd_visited[meeting]
                bwd_path = list(reversed(bwd_visited[meeting]))
                full_path = fwd_path + bwd_path[1:]
                return steps, full_path, meeting

        # Backward step
        if bwd_queue:
            node = bwd_queue.popleft()
            path = bwd_visited[node]
            added_b = []
            rev_ops = list(reversed(operators))
            for op, nb in get_neighbors(node, obstacle, rev_ops):
                if nb not in bwd_visited:
                    bwd_visited[nb] = path + [nb]
                    bwd_queue.append(nb)
                    added_b.append(nb)
            meeting = check_meeting()
            steps.append({
                "direction": "BACKWARD",
                "expanded": node,
                "fwd_queue": list(fwd_queue),
                "bwd_queue": list(bwd_queue),
                "fwd_visited": set(fwd_visited.keys()),
                "bwd_visited": set(bwd_visited.keys()),
                "added": added_b,
                "meeting": meeting,
                "found": meeting is not None,
            })
            if meeting:
                fwd_path = fwd_visited[meeting]
                bwd_path = list(reversed(bwd_visited[meeting]))
                full_path = fwd_path + bwd_path[1:]
                return steps, full_path, meeting

    return steps, None, None

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_grid(IS, FS, obstacle, cost_grid, path=None, visited=None, current=None,
                fwd_visited=None, bwd_visited=None, frontier=None, show_cost=True):
    if visited is None:
        visited = set()
    if frontier is None:
        frontier = set()
    path_set = set(path) if path else set()

    html = """<style>
.grid-table { border-collapse: collapse; margin: 10px auto; }
.grid-cell {
    width: 90px; height: 80px; border: 2px solid #555;
    text-align: center; vertical-align: middle;
    font-family: monospace; border-radius: 6px;
}
.coord { font-size: 11px; color: #555; }
.label { font-size: 13px; font-weight: bold; }
.cost-badge { font-size: 11px; background: #eee; border-radius: 3px; padding: 1px 4px; }
</style><table class="grid-table">"""

    for r in range(1, 4):
        html += "<tr>"
        for c in range(1, 4):
            cell = (r, c)
            cost = cost_grid.get(cell, 1)

            # Determine color
            if cell == obstacle:
                bg = CELL_COLORS["OBSTACLE"]
                label = "⛔ OBSTACLE"
                text_color = "white"
            elif cell == current:
                bg = CELL_COLORS["CURRENT"]
                label = "🔍 Expand"
                text_color = "white"
            elif fwd_visited and bwd_visited and cell in fwd_visited and cell in bwd_visited:
                bg = "#7B1FA2"
                label = "✨ MEET"
                text_color = "white"
            elif fwd_visited and cell in fwd_visited:
                bg = "#1565C0"
                label = ""
                text_color = "white"
            elif bwd_visited and cell in bwd_visited:
                bg = "#4A148C"
                label = ""
                text_color = "white"
            elif cell in path_set and cell != IS and cell != FS:
                bg = CELL_COLORS["PATH"]
                label = "📍"
                text_color = "white"
            elif cell in frontier:
                bg = CELL_COLORS["FRONTIER"]
                label = ""
                text_color = "#333"
            elif cell in visited and cell != IS and cell != FS:
                bg = CELL_COLORS["VISITED"]
                label = ""
                text_color = "#333"
            elif cell == IS:
                bg = CELL_COLORS["IS"]
                label = "🚀 IS"
                text_color = "white"
            elif cell == FS:
                bg = CELL_COLORS["FS"]
                label = "🏁 FS"
                text_color = "white"
            else:
                bg = CELL_COLORS["DEFAULT"]
                label = ""
                text_color = "#333"

            cost_display = f'<span class="cost-badge">cost={cost}</span>' if show_cost and cell != obstacle else ""
            html += f'<td class="grid-cell" style="background:{bg};color:{text_color};">'
            html += f'<div class="coord">({r},{c})</div>'
            html += f'<div class="label">{label}</div>'
            html += f'{cost_display}</td>'
        html += "</tr>"
    html += "</table>"
    return html

def fmt_cell(cell):
    return f"({cell[0]},{cell[1]})"

def fmt_list(cells):
    return "[" + ", ".join(fmt_cell(c) for c in cells) + "]"

def fmt_set(cells):
    return "{" + ", ".join(fmt_cell(c) for c in sorted(cells)) + "}"

def fmt_pq(pq_items):
    return "[" + ", ".join(f"({fmt_cell(c)},g={g})" for g, c in pq_items[:6]) + (", ..." if len(pq_items)>6 else "") + "]"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def check_step_answer(algo, step_data, step_idx, user_expanded, user_frontier, user_visited):
    correct_expanded = fmt_cell(step_data["expanded"])
    hints = []
    score = 0

    # Check expanded node
    if user_expanded.strip().replace(" ","") == correct_expanded.replace(" ",""):
        score += 1
    else:
        hints.append(f"❌ Node Diekspansi: Anda menulis **{user_expanded}**, seharusnya **{correct_expanded}**")

    # Check frontier (queue/stack)
    if algo in ("BFS",):
        correct_f = step_data.get("queue", [])
        correct_f_str = fmt_list(correct_f)
    elif algo == "DFS":
        correct_f = step_data.get("stack", [])
        correct_f_str = fmt_list(correct_f)
    elif algo == "UCS":
        correct_f = step_data.get("pq", [])
        correct_f_str = fmt_pq(correct_f)
    else:
        correct_f_str = ""

    # Loose check: contains right nodes?
    correct_nodes = {fmt_cell(c) if isinstance(c,tuple) else fmt_cell(c[1]) for c in correct_f} if correct_f else set()
    print(f"DEBUG: correct frontier nodes = {correct_nodes}")
    user_nodes = set(re.findall(r'\(\d+,\d+\)', user_frontier.replace(" ", "")))
    print(f"DEBUG: user frontier nodes = {user_nodes}")
    user_nodes = {u for u in user_nodes if u}
    print(f"DEBUG: user frontier nodes = {user_nodes}")


    if correct_nodes == user_nodes or not user_frontier.strip():
        if not user_frontier.strip() and correct_nodes:
            hints.append(f"⚠️ Frontier kosong — seharusnya **{correct_f_str}**")
        else:
            score += 1
    else:
        missing = correct_nodes - user_nodes
        extra = user_nodes - correct_nodes
        msg = f"⚠️ Frontier kurang akurat. "
        if missing:
            msg += f"Kurang: **{', '.join(missing)}**. "
        if extra:
            msg += f"Lebih: **{', '.join(extra)}**."
        hints.append(msg)

    return score, hints

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Blind Search Explorer")
    st.markdown("*LKS Kecerdasan Buatan*")
    st.divider()

    nim_input = st.text_input(
        "📋 Masukkan 5 Digit Terakhir NIM",
        max_chars=5,
        placeholder="Contoh: 12345",
        help="Parameter soal Anda akan digenerate otomatis berdasarkan NIM"
    )

    if nim_input and len(nim_input) == 5 and nim_input.isdigit():
        params = generate_params(nim_input)
        st.success(f"✅ Parameter untuk NIM ...{nim_input}")

        st.markdown("### 📌 Parameter Soal Anda")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Initial State", fmt_cell(params["IS"]))
            st.metric("Obstacle", fmt_cell(params["OBSTACLE"]))
        with col2:
            st.metric("Final State", fmt_cell(params["FS"]))

        st.markdown("**Urutan Operator:**")
        op_str = " → ".join(params["operators"])
        st.code(op_str, language=None)

        st.markdown("**Cost Grid:**")
        cg = params["cost_grid"]
        grid_md = "| | Kol 1 | Kol 2 | Kol 3 |\n|---|---|---|---|\n"
        for r in range(1,4):
            row_s = f"| **Baris {r}** |"
            for c in range(1,4):
                v = cg.get((r,c),1)
                cell_label = ""
                if (r,c) == params["IS"]: cell_label = " (IS)"
                elif (r,c) == params["FS"]: cell_label = " (FS)"
                elif (r,c) == params["OBSTACLE"]: cell_label = " (OBS)"
                row_s += f" {v}{cell_label} |"
            grid_md += row_s + "\n"
        st.markdown(grid_md)

    else:
        st.info("👆 Masukkan 5 digit terakhir NIM Anda untuk memulai")
        params = None

    st.divider()
    st.markdown("### ℹ️ Petunjuk")
    st.markdown("""
1. Masukkan NIM di atas
2. **Kerjakan LKS secara manual dulu!**
3. Gunakan tab **Validator** untuk cek jawaban
4. Gunakan tab **Visualisasi** untuk memahami alur
5. Tab **Perbandingan** setelah semua selesai
""")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if not params:
    st.markdown("# 🤖 Blind Search Explorer")
    st.markdown("### Selamat Datang!")
    st.info("👈 Masukkan 5 digit terakhir NIM Anda di sidebar untuk mendapatkan parameter soal unik Anda.")

    st.markdown("---")
    st.markdown("### 📚 Algoritma yang Dipelajari")
    cols = st.columns(3)
    algos_info = [
        ("BFS", "Breadth-First Search", "Menjelajah per level. Optimal untuk jalur terpendek.", "#1565C0"),
        ("DFS", "Depth-First Search", "Menjelajah sedalam mungkin. Hemat memori.", "#B71C1C"),
        ("DLS", "Depth-Limited Search", "DFS dengan batas kedalaman.", "#E65100"),
        ("IDS", "Iterative Deepening", "DLS berulang. Optimal + hemat memori.", "#1B5E20"),
        ("UCS", "Uniform Cost Search", "Ekspansi berdasarkan cost. Optimal untuk biaya.", "#4A148C"),
        ("BDS", "Bidirectional Search", "Pencarian dua arah. Sangat efisien.", "#006064"),
    ]
    for i, (short, full, desc, color) in enumerate(algos_info):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:{color};color:white;padding:15px;border-radius:10px;margin:5px 0;min-height:120px;">
<h4 style="margin:0;color:white;">{short}</h4>
<p style="font-size:12px;margin:4px 0;opacity:0.9;">{full}</p>
<p style="font-size:11px;margin:0;">{desc}</p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Grid & Parameter",
    "🔍 Validator Jawaban",
    "📊 Perbandingan",
    "📖 Referensi"
])

IS = params["IS"]
FS = params["FS"]
obstacle = params["OBSTACLE"]
operators = params["operators"]
cost_grid = params["cost_grid"]

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — GRID & PARAMETER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 🗺️ Grid & Parameter Soal")
    st.markdown(f"**NIM ...{nim_input}** — Salin informasi di bawah ke LKS Anda")

    col_grid, col_info = st.columns([1, 1])

    with col_grid:
        st.markdown("### Grid Gudang")
        grid_html = render_grid(IS, FS, obstacle, cost_grid, show_cost=True)
        st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown("**Legenda:**")
        st.markdown(f"""
- 🟢 **Hijau** = IS (Initial State / Start)
- 🟣 **Ungu** = FS (Final State / Goal)
- 🔴 **Merah** = Obstacle (tidak bisa dilewati)
- Angka `cost=N` = biaya memasuki sel tersebut
""")

    with col_info:
        st.markdown("### 📋 Ringkasan Parameter")

        st.markdown(f"""
| Parameter | Nilai |
|-----------|-------|
| **Initial State (IS)** | {fmt_cell(IS)} |
| **Final State (FS)** | {fmt_cell(FS)} |
| **Obstacle** | {fmt_cell(obstacle)} |
| **Urutan Operator** | {' → '.join(operators)} |
""")

        st.markdown("### 💰 Cost Tiap Cell")
        st.markdown("*Biaya saat robot MEMASUKI cell tersebut (untuk UCS)*")

        cost_rows = ""
        for r in range(1,4):
            cost_rows += f"| Baris {r} |"
            for c in range(1,4):
                v = cost_grid.get((r,c),1)
                tag = ""
                if (r,c)==IS: tag=" 🚀"
                elif (r,c)==FS: tag=" 🏁"
                elif (r,c)==obstacle: tag=" ⛔"
                cost_rows += f" **{v}**{tag} |"
            cost_rows += "\n"

        st.markdown(f"""
|  | Kol 1 | Kol 2 | Kol 3 |
|--|-------|-------|-------|
{cost_rows}""")

        st.markdown("### 🎯 Catatan Soal")
        st.info(f"""
Jalur dari **{fmt_cell(IS)}** ke **{fmt_cell(FS)}** \
menghindari obstacle di **{fmt_cell(obstacle)}**.

Urutan operator: **{' > '.join(operators)}**
(artinya jika ada beberapa neighbor valid, prioritas ekspansi mengikuti urutan ini)
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🔍 Validator Jawaban")
    st.warning("⚠️ **Kerjakan LKS secara manual terlebih dahulu!** Validator ini hanya untuk mengecek jawaban, bukan untuk contekan.")

    algo_choice = st.selectbox(
        "Pilih Algoritma yang Ingin Divalidasi:",
        ["BFS", "DFS", "DLS", "IDS", "UCS", "BDS"],
        help="Pilih setelah selesai mengerjakan bagian tersebut di LKS"
    )

    if algo_choice == "DLS":
        dls_limit = st.number_input("Depth Limit (L):", min_value=0, max_value=9, value=3)
    if algo_choice == "IDS":
        st.info("IDS divalidasi per iterasi. Pilih iterasi yang ingin dicek.")
        ids_iter = st.number_input("Iterasi ke- (Depth Limit = ?):", min_value=0, max_value=9, value=0)

    # Compute answer
    @st.cache_data
    def compute_bfs(is_, fs_, obs, ops):
        return bfs_steps(is_, fs_, obs, ops)

    @st.cache_data
    def compute_dfs(is_, fs_, obs, ops):
        return dfs_steps(is_, fs_, obs, ops)

    @st.cache_data
    def compute_dls(is_, fs_, obs, ops, lim):
        return dls_steps(is_, fs_, obs, ops, lim)

    @st.cache_data
    def compute_ids(is_, fs_, obs, ops):
        return ids_steps(is_, fs_, obs, ops)

    @st.cache_data
    def compute_ucs(is_, fs_, obs, ops, cg_json):
        cg = json.loads(cg_json)
        cg_tuple = {tuple(map(int, k.strip('()').split(','))): v for k,v in cg.items()}
        return ucs_steps(is_, fs_, obs, ops, cg_tuple)

    @st.cache_data
    def compute_bds(is_, fs_, obs, ops):
        return bds_steps(is_, fs_, obs, ops)

    IS_t = tuple(IS)
    FS_t = tuple(FS)
    OBS_t = tuple(obstacle)
    OPS_t = tuple(operators)
    CG_json = json.dumps({str(k): v for k,v in cost_grid.items()})

    if algo_choice == "BFS":
        steps, sol_path = compute_bfs(IS_t, FS_t, OBS_t, OPS_t)
        frontier_label = "Queue (FIFO)"
    elif algo_choice == "DFS":
        steps, sol_path = compute_dfs(IS_t, FS_t, OBS_t, OPS_t)
        frontier_label = "Stack (LIFO)"
    elif algo_choice == "DLS":
        steps, sol_path = compute_dls(IS_t, FS_t, OBS_t, OPS_t, dls_limit)
        frontier_label = "Stack"
    elif algo_choice == "IDS":
        all_iters = compute_ids(IS_t, FS_t, OBS_t, OPS_t)
        iter_idx = min(ids_iter, len(all_iters)-1)
        steps = all_iters[iter_idx]["steps"]
        sol_path = all_iters[iter_idx]["path"]
        frontier_label = "Stack"
    elif algo_choice == "UCS":
        steps, sol_path, sol_cost = compute_ucs(IS_t, FS_t, OBS_t, OPS_t, CG_json)
        frontier_label = "Priority Queue [(node,g),...]"
    else:  # BDS
        steps, sol_path, meeting_node = compute_bds(IS_t, FS_t, OBS_t, OPS_t)
        frontier_label = "Queue_F / Queue_B"

    total_steps = len(steps)

    st.markdown(f"**{algo_choice}** menghasilkan **{total_steps} langkah ekspansi**")

    # Step selector
    st.markdown("---")
    st.markdown("### Masukkan Jawaban Per Langkah")
    st.markdown("Pilih langkah, masukkan jawaban Anda, lalu klik **Cek Langkah Ini**.")

    step_num = st.slider("Langkah ke-:", 1, total_steps, 1)
    step_data = steps[step_num - 1]

    col_input, col_grid_v = st.columns([1, 1])

    with col_input:
        st.markdown(f"#### Langkah {step_num} dari {total_steps}")

        user_expanded = st.text_input(
            "Node yang Diekspansi (format: (r,c)):",
            placeholder="Contoh: (1,1)",
            key=f"exp_{algo_choice}_{step_num}"
        )

        if algo_choice == "UCS":
            user_frontier = st.text_input(
                f"Priority Queue [(node,g),...] setelah langkah ini:",
                placeholder="Contoh: [(1,2),g=1], [(2,1),g=2]",
                key=f"frt_{algo_choice}_{step_num}"
            )
        elif algo_choice == "BDS":
            user_frontier = st.text_input(
                "Queue_F setelah langkah ini (jika langkah FORWARD):",
                placeholder="Contoh: (1,2), (1,3)",
                key=f"frt_{algo_choice}_{step_num}"
            )
        else:
            user_frontier = st.text_input(
                f"{frontier_label} setelah langkah ini:",
                placeholder="Contoh: (1,2), (2,1), (1,3)",
                key=f"frt_{algo_choice}_{step_num}"
            )

        user_visited = st.text_input(
            "Visited setelah langkah ini:",
            placeholder="Contoh: (1,1), (1,2)",
            key=f"vis_{algo_choice}_{step_num}"
        )

        if st.button("✅ Cek Langkah Ini", type="primary"):
            score, hints = check_step_answer(
                algo_choice, step_data, step_num-1,
                user_expanded, user_frontier, user_visited
            )
            if score == 2:
                st.success("🎉 **Node Diekspansi dan Frontier BENAR!**")
            elif score == 1:
                st.warning("⚠️ Sebagian benar. Perhatikan hint berikut:")
            else:
                st.error("❌ Jawaban belum tepat. Perhatikan hint berikut:")

            for h in hints:
                st.markdown(h)

            if hints:
                st.info("💡 Kembali ke LKS dan perbaiki, lalu cek lagi. Jangan langsung lihat jawaban!")

        st.markdown("---")
        # Reveal button with confirmation
        if st.checkbox(f"🔓 Tampilkan jawaban langkah {step_num} (pastikan sudah mengerjakan manual!)"):
            st.markdown("**Jawaban Langkah ini:**")
            st.code(f"Node Diekspansi: {fmt_cell(step_data['expanded'])}")

            if algo_choice == "BFS":
                st.code(f"Queue: {fmt_list(step_data.get('queue',[]))}")
                st.code(f"Visited: {fmt_set(step_data.get('visited',set()))}")
            elif algo_choice == "DFS":
                st.code(f"Stack: {fmt_list(step_data.get('stack',[]))}")
                st.code(f"Visited: {fmt_set(step_data.get('visited',set()))}")
            elif algo_choice in ("DLS","IDS"):
                st.code(f"Depth: {step_data.get('depth',0)}")
                st.code(f"Stack: {fmt_list(step_data.get('stack',[]))}")
                st.code(f"Cutoff: {step_data.get('cutoff',False)}")
            elif algo_choice == "UCS":
                st.code(f"g(n): {step_data.get('g',0)}")
                st.code(f"Priority Queue: {fmt_pq(step_data.get('pq',[]))}")
            elif algo_choice == "BDS":
                st.code(f"Arah: {step_data.get('direction','')}")
                st.code(f"Queue_F: {fmt_list(step_data.get('fwd_queue',[]))}")
                st.code(f"Queue_B: {fmt_list(step_data.get('bwd_queue',[]))}")

    with col_grid_v:
        st.markdown("#### Visualisasi Grid Langkah Ini")

        visited_set = step_data.get("visited", set())
        current_node = step_data.get("expanded")
        if isinstance(visited_set, dict):
            visited_set = set(visited_set.keys())

        if algo_choice == "BDS":
            grid_html = render_grid(
                IS, FS, obstacle, cost_grid,
                current=current_node,
                fwd_visited=step_data.get("fwd_visited", set()),
                bwd_visited=step_data.get("bwd_visited", set()),
                show_cost=False
            )
        else:
            grid_html = render_grid(
                IS, FS, obstacle, cost_grid,
                current=current_node,
                visited=visited_set,
                show_cost=False
            )
        st.markdown(grid_html, unsafe_allow_html=True)

        if step_data.get("found"):
            st.success(f"🏁 **FS Ditemukan!** pada langkah ke-{step_num}")
            if sol_path:
                st.markdown(f"**Jalur:** {' → '.join(fmt_cell(c) for c in sol_path)}")

    # Final answer section
    st.markdown("---")
    st.markdown("### 📝 Cek Hasil Akhir")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        user_path = st.text_input(
            "Jalur yang Anda temukan:",
            placeholder="Contoh: (1,1) -> (1,2) -> (2,2)",
            key=f"path_{algo_choice}"
        )
    with col_r2:
        user_steps = st.number_input("Jumlah langkah:", min_value=0, value=0, key=f"steps_{algo_choice}")

    if st.button("Cek Hasil Akhir", key=f"check_final_{algo_choice}"):
        if sol_path:
            correct_path_str = " -> ".join(fmt_cell(c) for c in sol_path)
            correct_steps = len(sol_path) - 1
            st.markdown(f"**Jalur benar:** `{correct_path_str}`")
            st.markdown(f"**Jumlah langkah benar:** `{correct_steps}`")

            if algo_choice == "UCS" and sol_cost is not None:
                st.markdown(f"**Total cost benar:** `{sol_cost}`")
        else:
            st.warning("Tidak ada jalur yang ditemukan oleh algoritma ini (GAGAL / CUTOFF).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PERBANDINGAN
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Perbandingan Semua Algoritma")
    st.info("💡 Gunakan tab ini setelah Anda selesai mengerjakan SEMUA bagian di LKS.")

    if st.button("🚀 Jalankan Semua Algoritma & Bandingkan", type="primary"):
        results = {}

        with st.spinner("Menjalankan BFS..."):
            s, p = bfs_steps(IS_t, FS_t, OBS_t, OPS_t)
            results["BFS"] = {"steps": s, "path": p, "cost": len(p)-1 if p else None, "nodes": len(s)}

        with st.spinner("Menjalankan DFS..."):
            s, p = dfs_steps(IS_t, FS_t, OBS_t, OPS_t)
            results["DFS"] = {"steps": s, "path": p, "cost": len(p)-1 if p else None, "nodes": len(s)}

        with st.spinner("Menjalankan DLS (limit=4)..."):
            s, p = dls_steps(IS_t, FS_t, OBS_t, OPS_t, 4)
            results["DLS"] = {"steps": s, "path": p, "cost": len(p)-1 if p else None, "nodes": len(s)}

        with st.spinner("Menjalankan IDS..."):
            iters = ids_steps(IS_t, FS_t, OBS_t, OPS_t)
            total_nodes = sum(len(it["steps"]) for it in iters)
            final_path = iters[-1]["path"] if iters else None
            results["IDS"] = {"path": final_path, "cost": len(final_path)-1 if final_path else None,
                              "nodes": total_nodes, "iterations": len(iters)}

        with st.spinner("Menjalankan UCS..."):
            s, p, c = compute_ucs(IS_t, FS_t, OBS_t, OPS_t, CG_json)
            results["UCS"] = {"steps": s, "path": p, "cost": c, "nodes": len(s)}

        with st.spinner("Menjalankan BDS..."):
            s, p, m = bds_steps(IS_t, FS_t, OBS_t, OPS_t)
            results["BDS"] = {"steps": s, "path": p, "cost": len(p)-1 if p else None,
                              "nodes": len(s), "meeting": m}

        st.success("✅ Semua algoritma selesai dijalankan!")
        st.markdown("---")

        # Table
        st.markdown("### 📋 Tabel Perbandingan")
        table_md = "| Aspek | BFS | DFS | DLS | IDS | UCS | BDS |\n|---|---|---|---|---|---|---|\n"

        def pf(r): return " → ".join(fmt_cell(c) for c in r["path"]) if r["path"] else "GAGAL"
        def nf(r): return str(r.get("nodes","?"))
        def cf(r): return str(r.get("cost","?"))
        def uf(r): 
            if r.get("cost") is None: return "GAGAL"
            ucs_cost = results["UCS"].get("cost")
            return f"{'✅' if r['cost']==ucs_cost else '❌'} ({r['cost']} steps)"

        rows = [
            ("Jalur", {k: pf(v) for k,v in results.items()}),
            ("Node Diekspansi", {k: nf(v) for k,v in results.items()}),
            ("Langkah (jarak)", {k: cf(v) for k,v in results.items()}),
        ]

        for label, vals in rows:
            row = f"| **{label}** |"
            for algo in ["BFS","DFS","DLS","IDS","UCS","BDS"]:
                row += f" {vals.get(algo,'?')} |"
            table_md += row + "\n"

        table_md += "| **Optimal?** | ✅ Ya | ❌ Tidak | ❌ Tgt limit | ✅ Ya | ✅ Ya (cost) | ✅ Ya |\n"
        table_md += "| **Complete?** | ✅ Ya | ✅ Ya* | ⚠️ Jika L cukup | ✅ Ya | ✅ Ya | ✅ Ya |\n"
        st.markdown(table_md)

        # Bar chart: nodes expanded
        st.markdown("### 📈 Jumlah Node Diekspansi")
        node_counts = {k: v.get("nodes",0) for k,v in results.items()}
        import streamlit as st2
        st.bar_chart(node_counts)

        # Grid visualization per algo
        st.markdown("### 🗺️ Visualisasi Jalur Per Algoritma")
        cols_viz = st.columns(3)
        for i, (algo, res) in enumerate(results.items()):
            with cols_viz[i % 3]:
                color = ALGO_COLORS[algo]
                st.markdown(f"<div style='background:{color};color:white;padding:8px;border-radius:6px;text-align:center;font-weight:bold;'>{algo}</div>", unsafe_allow_html=True)
                grid_html = render_grid(IS, FS, obstacle, cost_grid,
                                        path=res.get("path"), show_cost=False)
                st.markdown(grid_html, unsafe_allow_html=True)
                if res["path"]:
                    st.caption(f"Jalur: {' → '.join(fmt_cell(c) for c in res['path'])}")
                else:
                    st.caption("Tidak ditemukan")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — REFERENSI
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 📖 Referensi Cepat Algoritma")

    for algo, color in ALGO_COLORS.items():
        with st.expander(f"**{algo}** — klik untuk buka", expanded=False):
            if algo == "BFS":
                st.markdown(f"""
**Struktur Data:** Queue (FIFO)
**Kompleksitas Waktu:** O(b^d) — b=branching factor, d=kedalaman solusi
**Kompleksitas Ruang:** O(b^d)
**Optimal:** ✅ Ya (jika semua biaya sama)
**Complete:** ✅ Ya (jika b terbatas)

**Sifat Kunci:**
- Semua node di level d diekspansi sebelum level d+1
- Menjamin jalur dengan **jumlah langkah terpendek**
- Membutuhkan banyak memori (semua node di frontier disimpan)
""")
            elif algo == "DFS":
                st.markdown(f"""
**Struktur Data:** Stack (LIFO)
**Kompleksitas Waktu:** O(b^m) — m=kedalaman maksimum
**Kompleksitas Ruang:** O(b*m) — jauh lebih hemat dari BFS!
**Optimal:** ❌ Tidak
**Complete:** ✅ Ya* (jika ada visited-set, tidak ada cycle tak terbatas)

**Sifat Kunci:**
- Menyelami satu jalur sedalam mungkin sebelum backtrack
- Bisa menemukan solusi lebih cepat jika solusi ada di jalur dalam
- Tidak optimal — jalur pertama yang ditemukan bukan tentu yang terpendek
""")
            elif algo == "DLS":
                st.markdown(f"""
**Struktur Data:** Stack (LIFO)
**Depth Limit:** L (parameter yang ditentukan pengguna)
**Optimal:** ❌ Tidak (kecuali L = kedalaman solusi optimal)
**Complete:** ⚠️ Hanya jika L ≥ kedalaman solusi

**Sifat Kunci:**
- DFS dengan batas kedalaman
- Jika node mencapai depth = L → CUTOFF (tidak diekspansi lebih lanjut)
- Mengatasi DFS yang bisa terjebak di jalur sangat panjang
- CUTOFF ≠ GAGAL: bisa jadi solusi ada tapi di luar limit
""")
            elif algo == "IDS":
                st.markdown(f"""
**Struktur Data:** Stack (LIFO), dijalankan berulang
**Kompleksitas Waktu:** O(b^d)
**Kompleksitas Ruang:** O(b*d) — seperti DFS!
**Optimal:** ✅ Ya (seperti BFS)
**Complete:** ✅ Ya

**Sifat Kunci:**
- Menggabungkan BFS (optimal, complete) + DFS (hemat memori)
- Node-node atas diekspansi ulang di setiap iterasi — trade-off yang sepadan
- Untuk pohon dengan branching factor b, overhead ekspansi ulang ≈ b/(b-1) ≈ kecil
""")
            elif algo == "UCS":
                st.markdown(f"""
**Struktur Data:** Priority Queue (min-heap berdasarkan g(n))
**Optimal:** ✅ Ya (selalu menemukan jalur dengan biaya terendah)
**Complete:** ✅ Ya

**Sifat Kunci:**
- g(n) = biaya kumulatif dari start ke node n
- Ekspansi berdasarkan **biaya masuk cell**, bukan jumlah langkah
- Jika semua biaya sama → identik dengan BFS
- Berbeda dari BFS ketika ada perbedaan biaya antar cell
""")
            elif algo == "BDS":
                st.markdown(f"""
**Struktur Data:** Dua Queue (forward dan backward)
**Kompleksitas Waktu:** O(b^(d/2)) — jauh lebih efisien!
**Optimal:** ✅ Ya (menggunakan BFS di kedua arah)

**Sifat Kunci:**
- Forward BFS dari IS, Backward BFS dari FS secara bersamaan
- Berhenti saat ada node yang dikunjungi KEDUANYA (titik pertemuan)
- Efektivitas: BFS normal = O(b^d); BDS = O(2×b^(d/2))
- Untuk d=6, b=3: BFS=729 node vs BDS=2×27=54 node!
""")

    st.markdown("---")
    st.markdown("### 🔑 Tabel Perbandingan Teoritis")
    st.markdown("""
| Algoritma | Complete? | Optimal? | Time | Space | Struktur |
|-----------|-----------|----------|------|-------|---------|
| BFS | ✅ | ✅ (uniform cost) | O(b^d) | O(b^d) | Queue |
| DFS | ✅* | ❌ | O(b^m) | O(bm) | Stack |
| DLS | ⚠️ | ❌ | O(b^L) | O(bL) | Stack |
| IDS | ✅ | ✅ | O(b^d) | O(bd) | Stack |
| UCS | ✅ | ✅ | O(b^(C*/ε)) | O(b^(C*/ε)) | PQueue |
| BDS | ✅ | ✅ | O(b^(d/2)) | O(b^(d/2)) | 2×Queue |

*b=branching factor, d=depth solusi, m=max depth, L=depth limit, C*=cost optimal, ε=min edge cost*
""")
