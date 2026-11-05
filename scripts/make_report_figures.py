"""Figures for report/report.tex, written to report/figures/ as vector PDF.

  fig_headline   paired AP difference vs the released FiDIP model with bootstrap 95% CIs
                 (results/report_data/bootstrap_vs_released.txt, from scripts/bootstrap_ap.py)
  fig_joints     share of joints missed on Test400, by joint and by visibility
                 (results/failure_analysis/, from scripts/failure_analysis.py)
  fig_skeletons  skeleton-only examples: label vs released vs fine-tuned, no SyRIP pixels
                 (reads private predictions and labels, so it needs output/ and data/)
Colours: slots 1 and 2 of the reference categorical palette; marker and line style also carry
identity, so the figures survive greyscale printing.
Usage: python scripts/make_report_figures.py
"""
import collections
import csv
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
OUT = 'report/figures'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    if os.environ.get('PNG_DIR'):
        fig.savefig(os.path.join(os.environ['PNG_DIR'], name + '.png'), bbox_inches='tight', dpi=130)


def headline():
    rows = collections.defaultdict(dict)
    for line in open('results/report_data/bootstrap_vs_released.txt'):
        kv = dict(re.findall(r'(\S+)=(\S+)', line))
        model = [k for k in kv if k not in ('subset', 'n_img', 'n_groups', 'released', 'diff', 'ci_lo', 'ci_hi', 'p_le0')][0]
        rows[model][kv['subset']] = (float(kv[model]) * 100, float(kv['diff']) * 100,
                                     float(kv['ci_lo']) * 100, float(kv['ci_hi']) * 100)
    order = [('ft40-s1-final', 'HRNet-W48 fine-tune, seed 1'), ('ft40-s2-final', 'HRNet-W48 fine-tune, seed 2'),
             ('sapiens2-1b-zs', 'Sapiens2-1B zero-shot'),
             ('vit2-h20-lr1e-5-s1-final', 'ViTPose++-H fine-tune, seed 1'),
             ('vit2-h20-lr1e-5-s2-final', 'ViTPose++-H fine-tune, seed 2'),
             ('ens-soup-s2-1b', 'ViTPose soup + Sapiens2 zero-shot'),
             ('full-s2ft-s1', 'Sapiens2-1B fine-tune, seed 1'), ('full-s2ft-s2', 'Sapiens2-1B fine-tune, seed 2')]
    subsets = [('test100', 'Test100 (primary)', 93.6), ('test400', 'Test400', 90.7),
               ('test400_indep', 'Test400, no source overlap', 90.2)]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75), sharey=True, sharex=True)
    y = np.arange(len(order))[::-1]
    for ax, (sub, title, base) in zip(axes, subsets):
        ax.axvline(0, color=MUTED, lw=1)
        for yy, (m, _) in zip(y, order):
            ap, d, lo, hi = rows[m][sub]
            final = m.startswith('full')
            c = BLUE if final else MUTED
            ax.plot([lo, hi], [yy, yy], color=c, lw=2 if final else 1.2, solid_capstyle='round')
            ax.plot(d, yy, 'o', color=c, ms=5.5 if final else 4.5, mec='white', mew=.8, zorder=3)
            ax.text(9.4, yy, f'{ap:.1f}', ha='right', va='center', fontsize=8, color=INK,
                    fontweight='bold' if final else 'normal')
        ax.set_title(f'{title}\nreleased FiDIP = {base}', fontsize=9, color=INK, loc='left')
        ax.set_xlim(-5, 9.6)
        ax.set_xticks([-4, 0, 4])
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n for _, n in order])
    for t, (m, _) in zip(axes[0].get_yticklabels(), order):
        if m.startswith('full'):
            t.set_fontweight('bold')
    axes[1].set_xlabel('AP difference from released FiDIP (points, 95% paired bootstrap CI); AP at right')
    fig.tight_layout(w_pad=.8)
    save(fig, 'fig_headline')


def joints():
    groups = [('Nose', ['nose']), ('Eyes', ['l_eye', 'r_eye']), ('Ears', ['l_ear', 'r_ear']),
              ('Shoulders', ['l_shoulder', 'r_shoulder']), ('Elbows', ['l_elbow', 'r_elbow']),
              ('Wrists', ['l_wrist', 'r_wrist']), ('Hips', ['l_hip', 'r_hip']),
              ('Knees', ['l_knee', 'r_knee']), ('Ankles', ['l_ankle', 'r_ankle'])]
    pj = [r for r in csv.DictReader(open('results/failure_analysis/per_joint.csv')) if r['subset'] == 'test400']
    vis = [r for r in csv.DictReader(open('results/failure_analysis/visibility.csv')) if r['subset'] == 'test400']

    def rate(rows, models, pred):
        sel = [r for r in rows if r['model'] in models and pred(r)]
        return 100 * sum(float(r['miss_rate']) * int(r['n']) for r in sel) / sum(int(r['n']) for r in sel)

    ft, rel = ('s2ft-s1', 's2ft-s2'), ('released',)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={'width_ratios': [2.1, 1]})

    def dumbbell(ax, labels, r, f):
        y = np.arange(len(labels))[::-1]
        for yy, x0, x1 in zip(y, r, f):
            ax.plot([x0, x1], [yy, yy], color=GRID, lw=2.5, solid_capstyle='round', zorder=1)
        ax.plot(r, y, 's', color=ORANGE, ms=6, mec='white', mew=.8, label='Released FiDIP', zorder=3)
        ax.plot(f, y, 'o', color=BLUE, ms=6.5, mec='white', mew=.8, label='Sapiens2-1B fine-tune', zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(left=0)
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    names = [g for g, _ in groups]
    r = [rate(pj, rel, lambda x, j=js: x['joint'] in j) for _, js in groups]
    f = [rate(pj, ft, lambda x, j=js: x['joint'] in j) for _, js in groups]
    dumbbell(a, names, r, f)
    a.set_title('By joint', fontsize=9, loc='left', color=INK)
    a.set_xlabel('Joints missed (%)')
    a.legend(frameon=False, loc='lower left', bbox_to_anchor=(.22, 1.0), ncol=2, fontsize=8, handletextpad=.2, borderaxespad=0)
    vs = ['visible', 'occluded']
    r = [rate(vis, rel, lambda x, v=v: x['visibility'] == v) for v in vs]
    f = [rate(vis, ft, lambda x, v=v: x['visibility'] == v) for v in vs]
    dumbbell(b, ['Visible\n(n=6452)', 'Occluded\n(n=340)'], r, f)
    b.set_ylim(-.6, 1.6)
    for yy, x0, x1 in zip([1, 0], r, f):
        b.text(x0 + 1.6, yy, f'{x0:.1f} to {x1:.1f}', va='center', fontsize=8, color=INK)
    b.set_xlim(0, 58)
    b.set_title('By label visibility', fontsize=9, loc='left', color=INK)
    b.set_xlabel('Joints missed (%)')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig_joints')


def skeletons():
    gt = json.load(open('data/syrip500/annotations/person_keypoints_validate_infant.json'))
    anns = {a['image_id']: a for a in gt['annotations']}

    def load(f):
        return {r['image_id']: np.array(r['keypoints']).reshape(17, 3)
                for r in json.load(open(f'output/audit12/results/{f}.json'))}

    rel, ft = load('released'), load('full-s2ft-s1')
    bones = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
             (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

    def oks(p, g, area):
        lab = g[:, 2] > 0
        d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
        return np.exp(-d ** 2 / (2 * area * (2 * SIGMAS) ** 2))[lab].mean()

    panels = [(229, 'Lying, arms raised'), (102, 'Side-lying, large toy'), (138, 'Upside-down, legs up'),
              (277, 'Seen from behind, sitting back'), (322, 'Prone, head towards camera'),
              (149, 'Likely label error')]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.5))
    for ax, (i, title) in zip(axes.ravel(), panels):
        g = np.array(anns[i]['keypoints'], dtype=float).reshape(17, 3)
        for k, c, ls, lw, mk, z in ((g, INK, '-', 2.4, 'o', 1), (rel[i], ORANGE, '--', 1.2, 's', 2),
                                    (ft[i], BLUE, '-', 1.2, 'o', 3)):
            for a, b in bones:
                ax.plot(k[[a, b], 0], k[[a, b], 1], color=c, ls=ls, lw=lw, alpha=.45 if c == INK else 1, zorder=z)
            ax.plot(k[:, 0], k[:, 1], mk, color=c, ms=3, mec='white', mew=.3, alpha=.6 if c == INK else 1, zorder=z)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f"{title}\nOKS: released {oks(rel[i], g, anns[i]['area']):.2f}, "
                     f"fine-tune {oks(ft[i], g, anns[i]['area']):.2f}", fontsize=8.5, color=INK, loc='left')
    handles = [plt.Line2D([], [], color=INK, lw=2.4, alpha=.45, marker='o', ms=3, label='Label'),
               plt.Line2D([], [], color=ORANGE, lw=1.2, ls='--', marker='s', ms=3, label='Released FiDIP'),
               plt.Line2D([], [], color=BLUE, lw=1.2, marker='o', ms=3, label='Sapiens2-1B fine-tune (seed 1)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, 'fig_skeletons')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    headline()
    joints()
    skeletons()
