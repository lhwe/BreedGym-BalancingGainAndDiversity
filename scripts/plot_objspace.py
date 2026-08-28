import os, sys, csv, math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import median

FIG = r'C:\Users\xxx\Desktop\abc'
OUT = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(os.path.dirname(os.path.abspath(__file__)), '_fig_objspace.png'))

# data
def load(fn):
    rows = list(csv.DictReader(open(os.path.join(FIG, fn), encoding='utf-8')))
    return [r for r in rows if r.get('Method', '').strip()]

def col(rows, name, key):
    return [float(r[key]) for r in rows if r['Method'] == name]

def q(v, p):
    v = sorted(v)
    if len(v) == 1: return v[0]
    k = (len(v) - 1) * p; f = math.floor(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

p1 = load('baseline_figure_terminal_results.csv')
p2 = load('reward_figure_terminal_results.csv')
p3 = load('obscomparison_terminal_results.csv')

rw   = [(float(r['Final Max GEBV']), float(r['HE Retention (%)']))
        for r in p2 if r['Method'].startswith('phase2_adjustw')]
fsib = [(float(r['Final Max GEBV']), float(r['HE Retention (%)']))
        for r in p2 if r['Method'] == 'Full-sib Penalty']
hyb  = [(float(r['Final Max GEBV']), float(r['HE Retention (%)']))
        for r in p2 if r['Method'] == 'Hybrid']

def big(rows, name):
    gx, hy = col(rows, name, 'Final Max GEBV'), col(rows, name, 'HE Retention (%)')
    return (median(gx), median(hy), q(gx, .1), q(gx, .9), q(hy, .1), q(hy, .9))

gs_x = col(p1, 'Standard GS', 'Final Max GEBV')[0]
gs_y = col(p1, 'Standard GS', 'HE Retention (%)')[0]
ga = big(p3, 'Generation-aware reference')
au = big(p3, 'Heterozygosity-augmented')
print(f'GS        : ({gs_x:.1f}, {gs_y:.1f})')
print(f'gen-aware : ({ga[0]:.1f}, {ga[1]:.1f})  whiskers x[{ga[2]:.1f},{ga[3]:.1f}] y[{ga[4]:.1f},{ga[5]:.1f}]')
print(f'augmented : ({au[0]:.1f}, {au[1]:.1f})  whiskers x[{au[2]:.1f},{au[3]:.1f}] y[{au[4]:.1f},{au[5]:.1f}]')

# convex hull (monotone chain)
def hull(pts):
    pts = sorted(set(pts))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    hi = []
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]

env = hull(rw + fsib + hyb)

def in_poly(pt, poly):
    x, y = pt; sign = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        c = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if c: 
            s = 1 if c > 0 else -1
            if sign and s != sign: return False
            sign = s
    return True

# plot
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.add_patch(plt.Polygon(env, closed=True, facecolor='0.72', edgecolor='none', alpha=0.45, zorder=1))
cx = sum(x for x, _ in env) / len(env); cy = sum(y for _, y in env) / len(env)
ax.annotate('reward tuning', (cx, cy + 2.5), ha='center', color='0.35', fontsize=10, zorder=2)

ax.scatter(*zip(*rw),   s=34, color='#F5A25F', label='heterozygosity-weighted rewards', zorder=3)
ax.scatter(*zip(*fsib), s=34, marker='s', color='#4DAF4A', label='full-sib penalty', zorder=3)
ax.scatter(*zip(*hyb),  s=34, marker='s', color='#9467BD', label='hybrid reward', zorder=3)
ax.scatter([gs_x], [gs_y], s=130, marker='D', color='k', label='GS', zorder=5)

for (m, c, lbl) in ((ga, '#C0392B', 'gen-aware'), (au, '#377EB8', 'augmented')):
    ax.errorbar(m[0], m[1], xerr=[[m[0]-m[2]], [m[3]-m[0]]], yerr=[[m[1]-m[4]], [m[5]-m[1]]],
                fmt='^', color=c, markersize=11, markeredgecolor='k', markerfacecolor=c,
                capsize=4, elinewidth=1.4, label=lbl, zorder=4)

ax.set_xlabel('Terminal maximum GEBV', fontsize=12)
ax.set_ylabel('Terminal expected heterozygosity', fontsize=12)
ax.grid(alpha=0.3, zorder=0)
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print('augmented marker outside reward envelope:', not in_poly((au[0], au[1]), env))
print('saved', OUT)
