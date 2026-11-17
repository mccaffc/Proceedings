"""Engraved field: a paper's theme drawn into the brand's tilted-line field, ink only (no scarlet).

Each cell carries one short stroke, as in the homepage field. Outside the motif the strokes follow the ground's slow
flow, faint and short. Inside it they turn to run along the form (the structure tensor of the tone study), the way an
engraver's lines follow a surface, and grow longer and heavier with the tone; the darkest tones take a crossing stroke.

  python3 scripts/illus/engrave.py SRC.png OUT.png [--style contour|snap] [--ground light|dark] [--w 2880 --h 1040]
"""
import argparse, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

PAL = {'light': ((0xE8, 0xE8, 0xE2), (0x2E, 0x32, 0x36)), 'dark': ((0x2E, 0x32, 0x36), (0xE8, 0xE8, 0xE2))}


def mulberry(a):
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


def flow_for(seed):
    R = mulberry(seed)
    terms = [(2 + 4 * R(), 2 + 4 * R(), 6.28 * R(), k) for k in (1, -1, 1, -2)]
    return lambda x, y: sum(math.sin(fx * x + fy * y + ph) for fx, fy, ph, _ in terms) * 0.55


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src'); p.add_argument('out')
    p.add_argument('--style', default='contour'); p.add_argument('--ground', default='light')
    p.add_argument('--w', type=int, default=2880); p.add_argument('--h', type=int, default=1040)
    p.add_argument('--cell', type=float, default=14); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--fit', type=float, default=1.0, help='motif height as a share of the band')
    p.add_argument('--x', type=float, default=0.62, help='motif centre as a share of the width')
    p.add_argument('--invert', action='store_true', help='lit surfaces carry the strokes (for light lines on a dark ground)')
    p.add_argument('--gain', type=float, default=1.0, help='scale the drawn tone (above 1 for a brighter drawing)')
    p.add_argument('--gamma', type=float, default=1.0, help='above 1 pushes the mid tones down, so only the strongest forms carry light')
    p.add_argument('--clear', action='append', default=[], help='x,y,w,h as shares of the band: a soft clearing for type, where the drawing drops to the ground')
    p.add_argument('--fade', type=float, default=0, help='fade the ground to nothing over this share of the band at its edges')
    p.add_argument('--bleed', action='store_true', help='the study runs off every edge: no feathered margin, paper measured from the top rows')
    p.add_argument('--field', help='also write the per-cell field as a PNG (R angle along the form, G weight, B tone, A ground) for live drawing')
    p.add_argument('--dump', help='also write the per-cell field (angle along the form, weight, tone) as JSON for live drawing')
    a = p.parse_args()
    W, H, cell = a.w, a.h, a.cell
    bg, ink = PAL[a.ground]

    # tone: 0 = paper, 1 = darkest; stretched per study, and placed in the band
    im = Image.open(a.src).convert('L')
    # lift the study's own paper (never quite white) to white, so its frame does not draw an edge in the field
    A = np.asarray(im, float)
    edge = A[:8].ravel() if a.bleed else np.concatenate([A[:8].ravel(), A[-8:].ravel(), A[:, :8].ravel(), A[:, -8:].ravel()])
    A = np.clip(A * 255 / max(np.percentile(edge, 90 if a.bleed else 60), 1), 0, 255)
    # and fade its margins to paper, so a form cut by the frame dissolves rather than ending in a straight line
    h, w = A.shape
    fy = np.clip(np.minimum(np.arange(h), h - 1 - np.arange(h)) / (0.12 * h), 0, 1)
    fx = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / (0.12 * w), 0, 1)
    F = np.outer(fy, fx)
    F = F * F * (3 - 2 * F)
    if a.bleed:
        F = np.ones_like(F)
    im = Image.fromarray(A.astype('uint8'))
    fim = Image.fromarray((F * 255).astype('uint8'))
    s = max(W / im.width, H / im.height) if a.bleed else a.fit * H / im.height
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    canvas = Image.new('L', (W, H), 255)
    fim = fim.resize(im.size, Image.LANCZOS)
    at = (round(a.x * W - im.width / 2), round((H - im.height) / 2))
    canvas.paste(im, at)
    fc = Image.new('L', (W, H), 0); fc.paste(fim, at)
    Fc = np.asarray(fc, float) / 255
    L = np.asarray(canvas, float) / 255
    lo, hi = np.percentile(L, 1), np.percentile(L, 99.5)
    T = np.clip((hi - L) / max(hi - lo, 1e-3), 0, 1)
    # the motif's footprint: anything not paper-white, slightly grown, so pale lit surfaces still read as form
    M = nd.gaussian_filter((L < hi - 0.02).astype(float), cell * 0.6)
    M = np.clip(M * 1.6, 0, 1)

    # orientation along the form: the minor eigenvector of the smoothed structure tensor
    Ts = nd.gaussian_filter(L, cell * 0.5)
    gy, gx = np.gradient(Ts)
    r = cell * 1.6
    Jxx, Jxy, Jyy = (nd.gaussian_filter(v, r) for v in (gx * gx, gx * gy, gy * gy))
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + math.pi / 2      # along the contour
    lam = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    coh = lam / (Jxx + Jyy + 1e-9)
    # edges: the engraver's outline, so a lit surface against the paper still has a contour
    E = nd.gaussian_filter(np.hypot(gx, gy), cell * 0.4)
    E = np.clip(E / max(np.percentile(E, 99.7), 1e-6), 0, 1)
    M = np.maximum(M, np.clip(E * 2, 0, 1)) * Fc

    clears = [tuple(map(float, c.split(','))) for c in a.clear]
    flow = flow_for(a.seed)
    SS = 3
    img = Image.new('RGB', (W * SS, H * SS), bg)
    d = ImageDraw.Draw(img)

    def seg(cx, cy, ang, length, width, col):
        h = cell * length / 2
        dx, dy = h * math.cos(ang), h * math.sin(ang)
        d.line([((cx - dx) * SS, (cy - dy) * SS), ((cx + dx) * SS, (cy + dy) * SS)], fill=col, width=max(1, round(width * SS)))

    def mix(t):
        return tuple(round(b + (i - b) * t) for b, i in zip(bg, ink))

    cols, rows = int(W // cell), int(H // cell)
    cells = []
    fld = np.zeros((rows, cols, 4), np.uint8) if a.field else None
    ox, oy = (W - cols * cell) / 2, (H - rows * cell) / 2
    for j in range(rows):
        for i in range(cols):
            cx, cy = ox + (i + 0.5) * cell, oy + (j + 0.5) * cell
            yi, xi = min(H - 1, int(cy)), min(W - 1, int(cx))
            m, t, c, e = M[yi, xi], T[yi, xi], coh[yi, xi], E[yi, xi]
            if a.invert:
                t = 1 - t
            t = t ** a.gamma
            ga = flow(cx / W * 2.2, cy / W * 2.2)                       # the ground's ordered flow
            fa = theta[yi, xi]
            w = m * min(1, c * 2.2)
            # take the shorter way round between the two directions (strokes are unoriented)
            dfa = (fa - ga + math.pi / 2) % math.pi - math.pi / 2
            ang = ga + dfa * w
            if a.style == 'snap':                                        # ascii: / | \ — and the half steps
                ang = round(ang / (math.pi / 8)) * (math.pi / 8)
            tone = m * min(1, (0.2 + 0.7 * t + 0.6 * e) * a.gain)
            for r in clears:                                              # the type's clearing
                dx = max(r[0] - cx / W, cx / W - r[0] - r[2], 0) * W / H
                dy = max(r[1] - cy / H, cy / H - r[1] - r[3], 0)
                q = min(1, math.hypot(dx, dy) / 0.06); tone *= 0.12 + 0.88 * q * q * (3 - 2 * q)
            ef = 1.0
            if a.fade:                                                    # the ground thins out toward the band's edges
                q = min(cx / W, 1 - cx / W) * H / W * 2.4, min(cy / H, 1 - cy / H)
                ef = min(1, min(q) / a.fade); ef = ef * ef * (3 - 2 * ef)
            length = 0.55 + 0.75 * tone
            width = cell * (0.06 + 0.2 * tone)
            col = mix(0.26 * ef + 0.6 * tone)
            seg(cx, cy, ang, length, width, col)
            if fld is not None:
                fld[j, i] = (round(fa % math.pi / math.pi * 255), round(w * 255), round(tone * 255), round(ef * 255))
            if a.dump:
                cells += [round(fa % math.pi * 1000), round(w * 255), round(tone * 255), round(ef * 255)]
    if fld is not None:
        Image.fromarray(fld, 'RGBA').save(a.field)
    if a.dump:
        import json
        json.dump({'cols': cols, 'rows': rows, 'cell': cell, 'w': W, 'h': H, 'seed': a.seed, 'cells': cells}, open(a.dump, 'w'), separators=(',', ':'))
    img.resize((W, H), Image.LANCZOS).save(a.out)


if __name__ == '__main__':
    main()
