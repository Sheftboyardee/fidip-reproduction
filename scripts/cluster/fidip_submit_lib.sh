# Shared submit helper: refuses to submit if it cannot read the job history (slow slurmctld
# returning nothing must not look like "never submitted"), and never retries.
C=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
submit() { tag=$1; shift; X=${EXCLUDE-c[2204-2207],d1025}
  hist=$(timeout 120 sacct -u $USER -S now-3days -X -n -P -o JobName,State) || { echo "$tag: sacct failed -- NOT submitting"; return 1; }
  [ -z "$hist" ] && { echo "$tag: empty job history -- NOT submitting"; return 1; }
  if echo "$hist" | grep -E "^fidip-$tag\|" | grep -qvE "\|(CANCELLED|FAILED)"; then echo "$tag: already submitted/ran, skipping"; return 0; fi
  id=$(sbatch --parsable -J fidip-$tag --time=${TIME:-01:00:00} ${X:+--exclude=$X} scripts/train.sbatch $C "$@" OUTPUT_DIR output/$tag LOG_DIR log/$tag 2>&1); echo "$tag: $id"; sleep 5; }
