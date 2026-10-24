"""Score one keypoint results json on Test500, Test400 (= Test500 minus Test100) and Test100.

model_best.pth is selected on Test100 by the training script, so Test400 is the unbiased
held-out number. Uses the same COCOeval settings as lib/dataset/syrip.py.
Usage: python scripts/subset_ap.py <gt500.json> <test100.json> <results.json> [tag]
Prints one CSV line: tag,set,AP,AP50,AP75,AR,AR50,AR75 for each subset.
"""
import contextlib
import io
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt500_file, gt100_file, res_file = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else res_file
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(gt500_file)
    dt = gt.loadRes(res_file)
ids500 = sorted(gt.getImgIds())
ids100 = {im['id'] for im in json.load(open(gt100_file))['images']}
subsets = {'test500': ids500,
           'test400': [i for i in ids500 if i not in ids100],
           'test100': [i for i in ids500 if i in ids100]}
for name, ids in subsets.items():
    ev = COCOeval(gt, dt, 'keypoints')
    ev.params.useSegm = None
    ev.params.imgIds = ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    print(f"{tag},{name},{s[0]:.4f},{s[1]:.4f},{s[2]:.4f},{s[5]:.4f},{s[6]:.4f},{s[7]:.4f}")
