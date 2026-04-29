"""
generate_diagram.py — Generate a polished model architecture diagram for V3.
Run: python generate_diagram.py
Outputs: model_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Color palette ─────────────────────────────────────────────────────────────
PAL = {
    'io':     ('#DBEAFE', '#2563EB', '#1E40AF'),   # face, edge, text
    'frozen': ('#DCFCE7', '#16A34A', '#14532D'),
    'proj':   ('#FEF3C7', '#D97706', '#78350F'),
    'trans':  ('#EDE9FE', '#7C3AED', '#4C1D95'),
    'norm':   ('#FEF9C3', '#CA8A04', '#713F12'),
    'head':   ('#FCE7F3', '#BE185D', '#831843'),
    'add':    ('#F8FAFC', '#94A3B8', '#334155'),
}

BG = '#FAFAFA'


def shadow(ax, cx, cy, w, h):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2 + 0.004, cy - h / 2 - 0.004), w, h,
        boxstyle='round,pad=0.013', facecolor='#00000018',
        edgecolor='none', linewidth=0, zorder=2))


def box(ax, cx, cy, w, h, text, sub='', ckey='io',
        fs=10.5, sfs=None, mono_sub=True):
    face, edge, tcol = PAL[ckey]
    sfs = sfs or fs - 2.0
    shadow(ax, cx, cy, w, h)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.013', facecolor=face,
        edgecolor=edge, linewidth=2.0, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.17, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#111', zorder=5)
        ax.text(cx, cy - h * 0.22, sub, ha='center', va='center',
                fontsize=sfs, color='#555', zorder=5,
                fontfamily='monospace' if mono_sub else 'sans-serif',
                style='normal')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#111', zorder=5)


def arr(ax, cx, y0, y1, lbl='', side='right', lfs=8):
    ax.annotate('', xy=(cx, y1 + 0.003), xytext=(cx, y0 - 0.003),
                arrowprops=dict(arrowstyle='->', color='#475569',
                                lw=2.0, mutation_scale=14),
                zorder=2)
    if lbl:
        x  = cx + 0.045 if side == 'right' else cx - 0.045
        ha = 'left'     if side == 'right' else 'right'
        ax.text(x, (y0 + y1) / 2, lbl, fontsize=lfs,
                color='#64748B', va='center', ha=ha, zorder=4)


def hline(ax, x0, x1, y, col='#94A3B8'):
    ax.plot([x0, x1], [y, y], color=col, lw=1.8, zorder=2)


def vline(ax, x, y0, y1, col='#94A3B8'):
    ax.plot([x, x], [y0, y1], color=col, lw=1.8, zorder=2)


def section_label(ax, cx, y, text, ckey='trans'):
    _, edge, _ = PAL[ckey]
    ax.text(cx, y, text, ha='center', va='center',
            fontsize=9, color=edge, fontweight='bold',
            fontstyle='italic', zorder=6)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=BG)
ax1 = fig.add_axes([0.02, 0.02, 0.50, 0.96])
ax2 = fig.add_axes([0.56, 0.03, 0.42, 0.94])

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis('off')

# Divider line
fig.add_artist(plt.Line2D([0.542, 0.542], [0.03, 0.97],
                           transform=fig.transFigure,
                           color='#CBD5E1', lw=1.5, linestyle='--'))


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT  —  Full model pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax1.text(0.50, 0.975, 'V3  Model Architecture',
         ha='center', va='top', fontsize=15, fontweight='bold', color='#1E293B')
ax1.text(0.50, 0.952, 'ResNet-50 Patch Embedder  +  Encoder-Only Transformer',
         ha='center', va='top', fontsize=9, color='#64748B', style='italic')

CX  = 0.50
BW  = 0.82

# Helper: returns (top, bottom) of a box given center and height
def tb(cy, h): return cy + h / 2, cy - h / 2

# ── Component definitions (cy, h) ────────────────────────────────────────────
IMG  = (0.893, 0.055)
RN   = (0.795, 0.100)     # ResNet container
FUS  = (0.667, 0.075)
CLS  = (0.568, 0.052)
PRJ  = (0.492, 0.052)
TRN  = (0.352, 0.148)     # transformer container
LNO  = (0.225, 0.052)
HD   = (0.143, 0.068)
ZS   = (0.058, 0.052)

# ── Input image ───────────────────────────────────────────────────────────────
box(ax1, CX, IMG[0], BW, IMG[1], 'Input Image', 'B × 3 × 224 × 224', 'io')
arr(ax1, CX, tb(*IMG)[1], tb(*RN)[0])

# ── ResNet-50 container ───────────────────────────────────────────────────────
rn_top, rn_bot = tb(*RN)
ax1.add_patch(FancyBboxPatch(
    (CX - BW / 2, rn_bot), BW, rn_top - rn_bot,
    boxstyle='round,pad=0.013', facecolor=PAL['frozen'][0],
    edgecolor=PAL['frozen'][1], linewidth=2.0, zorder=2))
ax1.text(CX, rn_top - 0.018, 'ResNet-50 Backbone  (frozen — ImageNet weights)',
         ha='center', va='center', fontsize=10.5, fontweight='bold', color='#14532D', zorder=4)

LW = 0.34
L_CY, L_H = RN[0] - 0.010, 0.045
box(ax1, CX - 0.22, L_CY, LW, L_H, 'Layer 3', 'B × 1024 × 14 × 14', 'frozen', fs=9)
box(ax1, CX + 0.22, L_CY, LW, L_H, 'Layer 4', 'B × 2048 × 7 × 7',   'frozen', fs=9)

mid_y = rn_bot + 0.016
ax1.annotate('', xy=(CX - 0.06, mid_y + 0.004),
             xytext=(CX - 0.22, L_CY - L_H / 2),
             arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.4), zorder=4)
ax1.annotate('', xy=(CX + 0.06, mid_y + 0.004),
             xytext=(CX + 0.22, L_CY - L_H / 2),
             arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.4), zorder=4)
ax1.text(CX, mid_y - 0.005,
         'AdaptiveAvgPool2d → Concat  ·  B × 3072 × 7 × 7',
         ha='center', va='center', fontsize=8, color='#166534', zorder=4)

arr(ax1, CX, rn_bot, tb(*FUS)[0])

# ── Feature fusion ────────────────────────────────────────────────────────────
box(ax1, CX, FUS[0], BW, FUS[1],
    '1×1 Conv  →  LayerNorm  →  Flatten + Transpose',
    'B × 3072 × 7 × 7   →   B × 49 × 768', 'proj')
arr(ax1, CX, tb(*FUS)[1], tb(*CLS)[0])

# ── CLS token ─────────────────────────────────────────────────────────────────
box(ax1, CX, CLS[0], BW, CLS[1],
    'Prepend Learned CLS Token   →   B × 50 × 768', ckey='proj')
arr(ax1, CX, tb(*CLS)[1], tb(*PRJ)[0])

# ── Input projection ──────────────────────────────────────────────────────────
box(ax1, CX, PRJ[0], BW, PRJ[1],
    'Linear Input Projection', 'd_patch = 768   →   d_model = 384', 'proj')
arr(ax1, CX, tb(*PRJ)[1], tb(*TRN)[0])

# ── Transformer block container ───────────────────────────────────────────────
tr_top, tr_bot = tb(*TRN)
ax1.add_patch(FancyBboxPatch(
    (CX - BW / 2, tr_bot), BW, tr_top - tr_bot,
    boxstyle='round,pad=0.013', facecolor=PAL['trans'][0],
    edgecolor=PAL['trans'][1], linewidth=2.5, linestyle='--', zorder=2))
ax1.text(CX, tr_top - 0.018, '× 6   Transformer Blocks',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color=PAL['trans'][2], zorder=4)

LINES = [
    ('Pre-LayerNorm   →   Multi-Head Self-Attention', True),
    ('8 heads  ·  head_dim = 48  ·  Dropout(0.2)  ·  Add residual', False),
    ('', False),
    ('Pre-LayerNorm   →   Feed-Forward Network', True),
    ('Linear(384→1536) → GELU → Dropout(0.2) → Linear(1536→384)  ·  Add residual', False),
]
for i, (txt, bold) in enumerate(LINES):
    if not txt:
        continue
    ax1.text(CX, tr_top - 0.048 - i * 0.023, txt,
             ha='center', va='center',
             fontsize=8.5 if bold else 7.8,
             fontweight='bold' if bold else 'normal',
             color='#4C1D95', zorder=4)

arr(ax1, CX, tr_bot, tb(*LNO)[0])

# ── Final LayerNorm + CLS ─────────────────────────────────────────────────────
box(ax1, CX, LNO[0], BW, LNO[1],
    'Final LayerNorm   →   Extract CLS Token', 'B × 384', 'norm')
arr(ax1, CX, tb(*LNO)[1], tb(*HD)[0])

# ── MLP head ──────────────────────────────────────────────────────────────────
box(ax1, CX, HD[0], BW, HD[1],
    'MLP Classification Head',
    'Linear(384→192)  →  GELU  →  Dropout(0.2)  →  Linear(192→10)', 'head')
arr(ax1, CX, tb(*HD)[1], tb(*ZS)[0])

# ── Z-score ───────────────────────────────────────────────────────────────────
box(ax1, CX, ZS[0], BW, ZS[1],
    'Z-Score Normalization   per image across aesthetics', ckey='norm')
arr(ax1, CX, tb(*ZS)[1], 0.008)

# ── Output ────────────────────────────────────────────────────────────────────
box(ax1, CX, 0.000, BW, 0.022,
    '10 Aesthetic Scores   [0 – 10]', ckey='io', fs=9.5)

# ── Legend ────────────────────────────────────────────────────────────────────
LEG = [
    ('Input / Output',       'io'),
    ('Frozen ResNet-50',     'frozen'),
    ('Learned Projections',  'proj'),
    ('Transformer Blocks',   'trans'),
    ('Normalization',        'norm'),
    ('Classification Head',  'head'),
]
lx, ly = 0.030, 0.188
ax1.text(lx, ly + 0.026, 'Legend', fontsize=9, fontweight='bold', color='#334155')
for i, (lbl, ckey) in enumerate(LEG):
    face, edge, _ = PAL[ckey]
    y = ly - i * 0.024
    ax1.add_patch(FancyBboxPatch(
        (lx, y - 0.008), 0.028, 0.016,
        boxstyle='round,pad=0.003', facecolor=face,
        edgecolor=edge, linewidth=1.2, zorder=3))
    ax1.text(lx + 0.036, y, lbl, fontsize=8, va='center', color='#334155')


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT  —  Transformer block detail
# ══════════════════════════════════════════════════════════════════════════════
ax2.text(0.50, 0.975, 'Transformer Block  ×6',
         ha='center', va='top', fontsize=14, fontweight='bold', color='#1E293B')
ax2.text(0.50, 0.950, 'Encoder-only  ·  Pre-LayerNorm  ·  No causal mask',
         ha='center', va='top', fontsize=9, color='#64748B', style='italic')

CX2 = 0.50
BW2 = 0.80

# Component positions
T = {}
T['inp']  = (0.885, 0.058)
T['ln1']  = (0.792, 0.058)
T['mha']  = (0.673, 0.090)
T['add1'] = (0.563, 0.058)
T['ln2']  = (0.475, 0.058)
T['ffn']  = (0.330, 0.108)
T['add2'] = (0.200, 0.058)
T['out']  = (0.105, 0.058)

box(ax2, CX2, T['inp'][0],  BW2, T['inp'][1],  'Input  x', 'B × 50 × 384', 'io')
arr(ax2, CX2, tb(*T['inp'])[1],  tb(*T['ln1'])[0])

box(ax2, CX2, T['ln1'][0],  BW2, T['ln1'][1],  'LayerNorm', ckey='norm')
arr(ax2, CX2, tb(*T['ln1'])[1],  tb(*T['mha'])[0])

box(ax2, CX2, T['mha'][0],  BW2, T['mha'][1],
    'Multi-Head Self-Attention',
    '8 heads  ·  head_dim = 48  ·  scale = 1/√48  ·  Dropout(0.2)', 'trans', fs=11)
arr(ax2, CX2, tb(*T['mha'])[1],  tb(*T['add1'])[0])

box(ax2, CX2, T['add1'][0], BW2, T['add1'][1], 'Add  &  Dropout  (Residual)', ckey='add')
arr(ax2, CX2, tb(*T['add1'])[1], tb(*T['ln2'])[0])

box(ax2, CX2, T['ln2'][0],  BW2, T['ln2'][1],  'LayerNorm', ckey='norm')
arr(ax2, CX2, tb(*T['ln2'])[1],  tb(*T['ffn'])[0])

box(ax2, CX2, T['ffn'][0],  BW2, T['ffn'][1],
    'Feed-Forward Network',
    'Linear(384 → 1536)  →  GELU  →  Dropout(0.2)  →  Linear(1536 → 384)',
    'proj', fs=11)
arr(ax2, CX2, tb(*T['ffn'])[1],  tb(*T['add2'])[0])

box(ax2, CX2, T['add2'][0], BW2, T['add2'][1], 'Add  &  Dropout  (Residual)', ckey='add')
arr(ax2, CX2, tb(*T['add2'])[1], tb(*T['out'])[0])

box(ax2, CX2, T['out'][0],  BW2, T['out'][1],  'Output  x', 'B × 50 × 384', 'io')

# ── Residual bypass 1: input → add1 (right) ──────────────────────────────────
RX1 = CX2 + BW2 / 2 + 0.065
inp_bot = tb(*T['inp'])[1]
add1_cy = T['add1'][0]
hline(ax2, CX2 + BW2 / 2, RX1, inp_bot)
vline(ax2, RX1, inp_bot, add1_cy)
ax2.annotate('', xy=(CX2 + BW2 / 2, add1_cy),
             xytext=(RX1, add1_cy),
             arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=1.8,
                             mutation_scale=12), zorder=2)
ax2.text(RX1 + 0.022, (inp_bot + add1_cy) / 2, 'skip\nconnection',
         fontsize=7.5, color='#64748B', va='center', ha='center',
         rotation=90, multialignment='center')

# ── Residual bypass 2: add1 → add2 (left) ────────────────────────────────────
RX2 = CX2 - BW2 / 2 - 0.065
add1_bot = tb(*T['add1'])[1]
add2_cy  = T['add2'][0]
hline(ax2, CX2 - BW2 / 2, RX2, add1_bot)
vline(ax2, RX2, add1_bot, add2_cy)
ax2.annotate('', xy=(CX2 - BW2 / 2, add2_cy),
             xytext=(RX2, add2_cy),
             arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=1.8,
                             mutation_scale=12), zorder=2)
ax2.text(RX2 - 0.022, (add1_bot + add2_cy) / 2, 'skip\nconnection',
         fontsize=7.5, color='#64748B', va='center', ha='center',
         rotation=90, multialignment='center')

# ── Design notes ──────────────────────────────────────────────────────────────
notes = (
    'Pre-LayerNorm applied before each sublayer (more stable gradients)\n'
    'No positional encoding — CNN patch tokens encode spatial info\n'
    'No causal mask — encoder classification, not autoregressive\n'
    'GELU activation in FFN  ·  Xavier uniform weight initialization'
)
ax2.text(0.50, 0.042, notes,
         ha='center', va='center', fontsize=8.5, color='#475569',
         style='italic', multialignment='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9',
                   edgecolor='#CBD5E1', linewidth=1.2))

plt.savefig('model_architecture.png', dpi=180, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print("Saved → model_architecture.png")
