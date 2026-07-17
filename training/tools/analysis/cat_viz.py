#!/usr/bin/env python3
"""
Interactive CAT move-order board visualiser.

Loads a random high-quality zeroink game and lets you step through it.
Board rendering matches ACEV13.html exactly (same palette, same geometry).

Controls:
  ← / → (or A / D)  — step backward / forward one ply
  R                  — pick a new random game
  Q                  — quit
"""
import json, pathlib, collections, random, sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# ── palette from ACEV13.html :root ────────────────────────────────────────────
BG      = '#1a1d23'
BOARD   = '#2b3038'
CELL_C  = '#3a414c'
WALL_C  = '#e8b04a'   # placed wall
P0_C    = '#5aa9e6'   # blue pawn
P1_C    = '#e66a6a'   # red pawn
GOOD_C  = '#7ec97e'   # green — benefits current player
BAD_C   = '#c95a5a'   # red   — hurts current player
CHOSEN_C= '#ffffff'   # white border = move actually played this ply
TXT_C   = '#d8dde6'
DIM_C   = '#8a93a3'

# ── board geometry (matches ACEV13 grid: 17×17, cell + groove interleaved) ────
#   cell (r, c) → top-left at (c*STEP, r*STEP) in data coords (y downward)
CELL   = 1.00
GROOVE = 0.22
STEP   = CELL + GROOVE      # 1.22 units per board slot
BSIZE  = 9 * STEP - GROOVE  # total board side length

# ── physics constants matching search.rs ──────────────────────────────────────
BASE       = 20_000
MAX_DETOUR = 6

# ── data source ───────────────────────────────────────────────────────────────
_repo = pathlib.Path(__file__).resolve().parents[3]   # repo root
ZEROINK = _repo / 'training' / 'data' / 'zeroink_games' / 'games_20260622.jsonl'
if not ZEROINK.exists():
    # fallback: hardcoded absolute path
    ZEROINK = pathlib.Path(r'c:\gitProjects\Quoridor best AI\training\data\zeroink_games\games_20260622.jsonl')

print('Loading games…', end=' ', flush=True)
ALL_RECORDS = [json.loads(l) for l in ZEROINK.read_text().splitlines()]
print(f'{len(ALL_RECORDS)} records')


# ─────────────────────────────────────────────────────────────────────────────
# BFS / scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_blocked(hw_list, vw_list):
    """Set of (min_cell, max_cell) pairs for every blocked edge."""
    b = set()
    for w in hw_list:
        x, y = w['x'], w['y']
        b.add((x*9+y,   (x+1)*9+y))
        b.add((x*9+y+1, (x+1)*9+y+1))
    for w in vw_list:
        x, y = w['x'], w['y']
        b.add((x*9+y,     x*9+y+1))
        b.add(((x+1)*9+y, (x+1)*9+y+1))
    return b

def bfs(sources, blocked):
    d = np.full(81, 255, dtype=np.int32)
    q = collections.deque()
    for s in sources:
        d[s] = 0; q.append(s)
    while q:
        u = q.popleft()
        r, c = divmod(u, 9)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                v = nr*9+nc
                if (min(u,v), max(u,v)) not in blocked and d[v] == 255:
                    d[v] = d[u]+1; q.append(v)
    return d

def compute_fields(state):
    bl = get_blocked(state['horizontalWalls'], state['verticalWalls'])
    d0 = bfs(range(72, 81), bl)   # P0 goal: row 8  (zeroink P0 races to row 8)
    d1 = bfs(range(0,  9),  bl)   # P1 goal: row 0
    cw0 = np.bincount(np.clip(d0, 0, 17), minlength=18).astype(np.int32)
    cw1 = np.bincount(np.clip(d1, 0, 17), minlength=18).astype(np.int32)
    zp0 = state['player0Cell']
    zp1 = state['player1Cell']
    pr0, pc0 = divmod(zp0, 9)
    pr1, pc1 = divmod(zp1, 9)
    return d0, d1, cw0, cw1, pr0, pc0, int(d0[zp0]), pr1, pc1, int(d1[zp1])

def cat_score(m, me, d0, d1, cw0, cw1, pr0, pc0, total0, pr1, pc1, total1):
    slot = m % 100
    a = (slot >> 3) * 9 + (slot & 7)
    edges = [(a, a+9), (a+1, a+10)] if m < 200 else [(a, a+1), (a+9, a+10)]
    score = 0
    for el, er in edges:
        for pid, d, cw, pr, pc, tot in [
            (0, d0, cw0, pr0, pc0, total0),
            (1, d1, cw1, pr1, pc1, total1),
        ]:
            dl, dr = int(d[el]), int(d[er])
            if dl != dr:
                entry = el if dl > dr else er
                ei_r, ei_c = divmod(entry, 9)
                manh   = abs(ei_r - pr) + abs(ei_c - pc)
                detour = manh + int(d[entry]) - tot
                w = max(0, MAX_DETOUR - detour)
                if w > 0:
                    layer = min(dl, dr)
                    width = max(int(cw[min(layer, 17)]), 1)
                    c = BASE * w // width
                    score += -c if me == pid else c
    return score

def gen_pawn_moves(pawn_me, pawn_opp, blocked):
    """All legal pawn moves including jumps."""
    moves = []
    pr, pc = divmod(pawn_me, 9)
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = pr+dr, pc+dc
        if not (0 <= nr < 9 and 0 <= nc < 9):
            continue
        dest = nr*9+nc
        if (min(pawn_me, dest), max(pawn_me, dest)) in blocked:
            continue
        if dest == pawn_opp:
            # straight jump
            nr2, nc2 = nr+dr, nc+dc
            jumped = False
            if 0 <= nr2 < 9 and 0 <= nc2 < 9:
                dest2 = nr2*9+nc2
                if (min(pawn_opp, dest2), max(pawn_opp, dest2)) not in blocked:
                    moves.append(dest2); jumped = True
            if not jumped:
                # diagonal jumps
                for ddr, ddc in ((-1,0),(1,0),(0,-1),(0,1)):
                    if (ddr, ddc) == (dr, dc) or (ddr, ddc) == (-dr, -dc):
                        continue
                    nr3, nc3 = nr+ddr, nc+ddc
                    if 0 <= nr3 < 9 and 0 <= nc3 < 9:
                        dest3 = nr3*9+nc3
                        if (min(pawn_opp, dest3), max(pawn_opp, dest3)) not in blocked:
                            moves.append(dest3)
        else:
            moves.append(dest)
    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers  (y increases downward; row 0 = top of board)
# ─────────────────────────────────────────────────────────────────────────────

def cell_xy(row, col):
    """Top-left of a cell rectangle."""
    return col * STEP, row * STEP

def hw_xywh(wx, wy):
    """(x, y, w, h) of a horizontal wall slot (wx=row-gap 0-7, wy=col-start 0-7)."""
    return wy * STEP, wx * STEP + CELL, 2*CELL + GROOVE, GROOVE

def vw_xywh(wx, wy):
    """(x, y, w, h) of a vertical wall slot (wx=row-start 0-7, wy=col-gap 0-7)."""
    return wy * STEP + CELL, wx * STEP, GROOVE, 2*CELL + GROOVE


# ─────────────────────────────────────────────────────────────────────────────
# Game loader
# ─────────────────────────────────────────────────────────────────────────────

def pick_game(rng=None):
    if rng is None:
        rng = random.Random()
    ids = list({r['game_id'] for r in ALL_RECORDS})
    rng.shuffle(ids)
    for gid in ids:
        g = sorted([r for r in ALL_RECORDS if r['game_id'] == gid],
                   key=lambda r: r['move_num'])
        last = g[-1]['state']
        if len(last['horizontalWalls']) + len(last['verticalWalls']) >= 5:
            return gid, g
    gid = ids[0]
    g = sorted([r for r in ALL_RECORDS if r['game_id'] == gid], key=lambda r: r['move_num'])
    return gid, g


# ─────────────────────────────────────────────────────────────────────────────
# Interactive board window
# ─────────────────────────────────────────────────────────────────────────────

class BoardViz:
    def __init__(self):
        self.fig = plt.figure(figsize=(13, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title('CAT Move-Order Visualiser')

        # Board axes (left, square)
        self.ax = self.fig.add_axes([0.02, 0.11, 0.68, 0.84])
        self.ax.set_facecolor(BOARD)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Info panel (right)
        self.ax_info = self.fig.add_axes([0.73, 0.11, 0.25, 0.84])
        self.ax_info.set_facecolor(BOARD)
        for sp in self.ax_info.spines.values():
            sp.set_visible(False)
        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])

        # Buttons
        kw = dict(color='#343b46', hovercolor='#505a6a')
        self.btn_prev = Button(self.fig.add_axes([0.07, 0.02, 0.12, 0.055]), '<< Prev', **kw)
        self.btn_next = Button(self.fig.add_axes([0.21, 0.02, 0.12, 0.055]), 'Next >>', **kw)
        self.btn_rand = Button(self.fig.add_axes([0.38, 0.02, 0.15, 0.055]), 'New game', **kw)
        for b in (self.btn_prev, self.btn_next, self.btn_rand):
            b.label.set_color(TXT_C); b.label.set_fontsize(10)

        self.btn_prev.on_clicked(lambda _: self.step(-1))
        self.btn_next.on_clicked(lambda _: self.step(+1))
        self.btn_rand.on_clicked(lambda _: self.new_game())
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.game_id = None
        self.game    = []
        self.frame   = 0
        self.new_game()

        # Force window to front (Tk-specific)
        try:
            win = self.fig.canvas.manager.window
            win.lift()
            win.attributes('-topmost', True)
            win.after(500, lambda: win.attributes('-topmost', False))
            win.focus_force()
        except Exception:
            pass

    # ── navigation ───────────────────────────────────────────────────────────

    def new_game(self):
        self.game_id, self.game = pick_game()
        self.frame = 0
        self.redraw()

    def step(self, delta):
        self.frame = max(0, min(len(self.game)-1, self.frame + delta))
        self.redraw()

    def _on_key(self, ev):
        if ev.key in ('right', 'd'): self.step(+1)
        elif ev.key in ('left',  'a'): self.step(-1)
        elif ev.key == 'r':            self.new_game()
        elif ev.key == 'q':            plt.close('all')

    # ── drawing ───────────────────────────────────────────────────────────────

    def redraw(self):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(BOARD)
        ax.set_xlim(-0.15, BSIZE + 0.15)
        ax.set_ylim(BSIZE + 0.15, -0.15)
        ax.set_aspect('equal')
        ax.axis('off')

        rec   = self.game[self.frame]
        state = rec['state']
        me    = state['currentPlayer']
        zp0   = state['player0Cell']
        zp1   = state['player1Cell']
        chosen = rec['move_chosen']

        bl = get_blocked(state['horizontalWalls'], state['verticalWalls'])
        d0, d1, _, _, _, _, total0, _, _, total1 = compute_fields(state)

        # ── from-pawn BFS: which cells are reachable optimally? ───────────────
        # fp0[c] = distance from pawn0 to cell c (respecting walls)
        # A cell c is on player 0's shortest-path DAG iff fp0[c] + d0[c] == total0
        fp0 = bfs([zp0], bl)
        fp1 = bfs([zp1], bl)

        on0 = np.array([int(fp0[c] + d0[c] == total0 and d0[c] < 200) for c in range(81)])
        on1 = np.array([int(fp1[c] + d1[c] == total1 and d1[c] < 200) for c in range(81)])

        # DAG corridor width per goal-distance layer (proper narrow bottleneck detection)
        dag_cw0 = np.zeros(18, dtype=int)
        dag_cw1 = np.zeros(18, dtype=int)
        for c in range(81):
            if on0[c] and d0[c] < 18: dag_cw0[d0[c]] += 1
            if on1[c] and d1[c] < 18: dag_cw1[d1[c]] += 1

        # ── edge importance: how critical is (u,v) to path structure ──────────
        # An edge is "on player p's DAG" iff going pawn→u→v→goal is optimal.
        # Weight by 1/corridor_width: bottleneck edges score high.
        def edge_imp(u, v):
            score = 0.0
            # P0
            if (fp0[u] + 1 + d0[v] == total0 and d0[v] < d0[u]) or \
               (fp0[v] + 1 + d0[u] == total0 and d0[u] < d0[v]):
                layer = min(int(d0[u]), int(d0[v]))
                score += 1.0 / max(dag_cw0[layer] if layer < 18 else 1, 1)
            # P1
            if (fp1[u] + 1 + d1[v] == total1 and d1[v] < d1[u]) or \
               (fp1[v] + 1 + d1[u] == total1 and d1[u] < d1[v]):
                layer = min(int(d1[u]), int(d1[v]))
                score += 1.0 / max(dag_cw1[layer] if layer < 18 else 1, 1)
            return score

        # ── wall slot importance ───────────────────────────────────────────────
        hw_placed = {(w['x'], w['y']) for w in state['horizontalWalls']}
        vw_placed = {(w['x'], w['y']) for w in state['verticalWalls']}
        hw_imp = {}; vw_imp = {}
        for wx in range(8):
            for wy in range(8):
                a = wx * 9 + wy
                if (wx, wy) not in hw_placed:
                    s = edge_imp(a, a+9) + edge_imp(a+1, a+10)
                    if s > 0: hw_imp[(wx, wy)] = s
                if (wx, wy) not in vw_placed:
                    s = edge_imp(a, a+1) + edge_imp(a+9, a+10)
                    if s > 0: vw_imp[(wx, wy)] = s

        all_imp = list(hw_imp.values()) + list(vw_imp.values())
        imp_max = max(all_imp) if all_imp else 1.0

        # ── pawn move ordering ─────────────────────────────────────────────────
        dist_me  = d0 if me == 0 else d1
        total_me = total0 if me == 0 else total1
        pawn_me  = zp0 if me == 0 else zp1
        pawn_opp = zp1 if me == 0 else zp0
        legal_pawns = gen_pawn_moves(pawn_me, pawn_opp, bl)
        pawn_scores = {dest: 1_000_000 - int(dist_me[dest]) * 1000 for dest in legal_pawns}
        if pawn_scores:
            ps_max = max(pawn_scores.values())
            pawn_n = {d: s / ps_max for d, s in pawn_scores.items()}
        else:
            pawn_n = {}
        pawn_ranked = sorted(pawn_n, key=lambda d: pawn_n[d], reverse=True)

        RADIUS = 0.08

        # ── 1. base cells ─────────────────────────────────────────────────────
        for row in range(9):
            for col in range(9):
                cx, cy = cell_xy(row, col)
                ax.add_patch(patches.FancyBboxPatch(
                    (cx, cy), CELL, CELL,
                    boxstyle=f'round,pad=0,rounding_size={RADIUS}',
                    facecolor=CELL_C, edgecolor='none', zorder=1,
                ))

        # ── 2. path-DAG cell heatmap ──────────────────────────────────────────
        # Blue tint = only P0's optimal path, Red tint = only P1's, White = both
        for c in range(81):
            a0, a1 = on0[c], on1[c]
            if not a0 and not a1: continue
            row, col = divmod(c, 9)
            cx, cy = cell_xy(row, col)
            if a0 and a1:
                color, alpha = '#ffffff', 0.20   # overlap = brightest
            elif a0:
                color, alpha = P0_C, 0.13
            else:
                color, alpha = P1_C, 0.13
            ax.add_patch(patches.FancyBboxPatch(
                (cx, cy), CELL, CELL,
                boxstyle=f'round,pad=0,rounding_size={RADIUS}',
                facecolor=color, edgecolor='none', alpha=alpha, zorder=2,
            ))

        # ── 3. wall importance: red intensity scale ────────────────────────────
        # transparent = no path impact; orange = medium; bright red = bottleneck
        for wall_dict, xywh_fn in ((hw_imp, hw_xywh), (vw_imp, vw_xywh)):
            for (wx, wy), imp in wall_dict.items():
                bx, by, bw, bh = xywh_fn(wx, wy)
                n = imp / imp_max
                # colour ramp: orange (#e07820) → red (#d02020)
                r = int(0xe0 - n * (0xe0 - 0xd0))
                g = int(0x78 * (1 - n))
                color = f'#{r:02x}{g:02x}20'
                alpha = 0.10 + n * 0.82
                ax.add_patch(patches.Rectangle(
                    (bx, by), bw, bh,
                    facecolor=color, edgecolor='none', alpha=alpha, zorder=3,
                ))

        # ── 4. placed walls: gold with bright white border ────────────────────
        for w in state['horizontalWalls']:
            bx, by, bw, bh = hw_xywh(w['x'], w['y'])
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh,
                facecolor=WALL_C, edgecolor='white', linewidth=1.5, zorder=5,
            ))
        for w in state['verticalWalls']:
            bx, by, bw, bh = vw_xywh(w['x'], w['y'])
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh,
                facecolor=WALL_C, edgecolor='white', linewidth=1.5, zorder=5,
            ))

        # ── 5. pawn move priority ─────────────────────────────────────────────
        for rank_i, dest in enumerate(pawn_ranked):
            n = pawn_n[dest]
            row, col = divmod(dest, 9)
            cx, cy = cell_xy(row, col)
            alpha = 0.20 + n * 0.65
            ax.add_patch(patches.FancyBboxPatch(
                (cx, cy), CELL, CELL,
                boxstyle=f'round,pad=0,rounding_size={RADIUS}',
                facecolor='#b0e0ff', edgecolor='#ffffff',
                linewidth=1.0, alpha=alpha, zorder=6,
            ))
            ax.text(cx + CELL/2, cy + CELL/2, str(rank_i + 1),
                    ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold', zorder=11)

        # ── 6. move played (white border) ─────────────────────────────────────
        if chosen['kind'] == 'pawn':
            row, col = divmod(chosen['target'], 9)
            cx, cy = cell_xy(row, col)
            ax.add_patch(patches.FancyBboxPatch(
                (cx, cy), CELL, CELL,
                boxstyle=f'round,pad=0,rounding_size={RADIUS}',
                facecolor='none', edgecolor=CHOSEN_C, linewidth=2.5, zorder=12,
            ))
        elif chosen['kind'] == 'wall':
            wx_c, wy_c = chosen['x'], chosen['y']
            is_hw = chosen['orientation'] == 'horizontal'
            bx, by, bw, bh = hw_xywh(wx_c, wy_c) if is_hw else vw_xywh(wx_c, wy_c)
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh,
                facecolor='none', edgecolor=CHOSEN_C, linewidth=3.0, zorder=12,
            ))

        # ── 7. pawns ──────────────────────────────────────────────────────────
        for cell, color in ((zp0, P0_C), (zp1, P1_C)):
            row, col = divmod(cell, 9)
            cx, cy = cell_xy(row, col)
            ox, oy = cx + CELL/2, cy + CELL/2
            ax.add_patch(patches.Circle((ox, oy+0.05), CELL*0.37,
                facecolor='#000', alpha=0.25, zorder=13))
            ax.add_patch(patches.Circle((ox, oy), CELL*0.37,
                facecolor=color, edgecolor='white', linewidth=1.2, zorder=14))
            ax.add_patch(patches.Circle((ox-CELL*0.11, oy-CELL*0.11), CELL*0.11,
                facecolor='white', alpha=0.45, zorder=15))

        # ── 8. info panel ─────────────────────────────────────────────────────
        ai = self.ax_info
        ai.cla(); ai.set_facecolor(BOARD)
        for sp in ai.spines.values(): sp.set_visible(False)
        ai.set_xticks([]); ai.set_yticks([])
        ai.set_xlim(0, 1); ai.set_ylim(0, 1)

        winner      = self.game[0]['game_outcome'].get('winner', '?')
        total_plies = len(self.game)

        def info_row(y, label, value, vc=TXT_C):
            ai.text(0.04, y, label, transform=ai.transAxes,
                    fontsize=9.5, color=DIM_C, va='top', fontfamily='monospace')
            ai.text(0.52, y, str(value), transform=ai.transAxes,
                    fontsize=9.5, color=vc, va='top', fontfamily='monospace')

        y = 0.97
        info_row(y, 'Game',    self.game_id);                       y -= 0.055
        info_row(y, 'Ply',     f'{self.frame+1}/{total_plies}');    y -= 0.055
        info_row(y, 'Move#',   rec['move_num']);                     y -= 0.055
        y -= 0.015
        info_row(y, 'To move', f'P{me}', P0_C if me==0 else P1_C); y -= 0.055
        info_row(y, 'P0 dist', f'{total0}', P0_C);                  y -= 0.055
        info_row(y, 'P1 dist', f'{total1}', P1_C);                  y -= 0.055
        info_row(y, 'P0 walls',state['player0Walls'], P0_C);        y -= 0.055
        info_row(y, 'P1 walls',state['player1Walls'], P1_C);        y -= 0.055
        y -= 0.015
        info_row(y, 'Winner',  f'P{winner}', P0_C if winner==0 else P1_C); y -= 0.055
        y -= 0.025

        def legend_box(yy, color, label, ec='none'):
            ai.add_patch(patches.Rectangle((0.04, yy-0.018), 0.09, 0.026,
                facecolor=color, edgecolor=ec, linewidth=1,
                transform=ai.transAxes, zorder=3))
            ai.text(0.17, yy, label, transform=ai.transAxes,
                    fontsize=8.0, color=TXT_C, va='center')

        ai.text(0.04, y, 'Cells', transform=ai.transAxes,
                fontsize=8.5, color=DIM_C, va='top'); y -= 0.048
        legend_box(y, '#ffffff', 'both paths (overlap)', ec='none'); y -= 0.044
        legend_box(y, P0_C,     'P0 path only');        y -= 0.044
        legend_box(y, P1_C,     'P1 path only');        y -= 0.044
        legend_box(y, '#b0e0ff', 'pawn move (1=best)'); y -= 0.052

        ai.text(0.04, y, 'Walls', transform=ai.transAxes,
                fontsize=8.5, color=DIM_C, va='top'); y -= 0.048
        legend_box(y, '#d02020', 'high path impact');   y -= 0.044
        legend_box(y, '#e07820', 'medium impact');      y -= 0.044
        legend_box(y, WALL_C,   'placed', ec='white');  y -= 0.044
        legend_box(y, 'none',   'move played', ec='white'); y -= 0.052

        ai.text(0.04, y, 'Keys', transform=ai.transAxes,
                fontsize=8.5, color=DIM_C, va='top'); y -= 0.044
        for key, desc in (('<- ->', 'step'), ('R', 'new game'), ('Q', 'quit')):
            ai.text(0.04, y, key, transform=ai.transAxes, fontsize=8,
                    color=TXT_C, va='top', fontfamily='monospace')
            ai.text(0.38, y, desc, transform=ai.transAxes, fontsize=8,
                    color=DIM_C, va='top')
            y -= 0.042

        self.fig.suptitle(
            f'Move-Order Visualiser  -  {self.game_id}  (ply {self.frame+1}/{total_plies})',
            color=TXT_C, fontsize=10, y=0.987,
        )
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    import traceback
    try:
        viz = BoardViz()
        plt.show()
    except Exception:
        traceback.print_exc()
        input('\nPress Enter to close...')
