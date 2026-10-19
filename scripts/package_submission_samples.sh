#!/usr/bin/env bash
# Collect the demo audio samples (phase_6/results/demo_samples, made by scripts/make_demo_samples.py) into
# one folder and zip it for the submission: WAVs, README, index.csv and PDF figures (PNGs left out: 3x larger,
# and the zip must stay well under the 25 MB e-mail limit once base64-encoded).
#
#   scripts/package_submission_samples.sh            # -> submission_samples/ and submission_samples.zip
#   scripts/package_submission_samples.sh my_name    # -> my_name/ and my_name.zip
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${repo_root}/phase_6/results/demo_samples"
name="${1:-submission_samples}"
dest="${repo_root}/${name}"

if [[ ! -f "${src}/index.csv" ]]; then
    echo "missing ${src}/index.csv: run git pull (or scripts/make_demo_samples.py) first" >&2
    exit 1
fi

rm -rf "${dest}" "${dest}.zip"
mkdir -p "${dest}/figures"
cp "${src}/README.md" "${src}/index.csv" "${dest}/"
cp "${src}"/figures/*.pdf "${dest}/figures/"
for gap_dir in "${src}"/gap_*/; do
    cp -r "${gap_dir%/}" "${dest}/"
done

cd "${repo_root}"
if command -v zip >/dev/null 2>&1; then
    zip -qr -X "${name}.zip" "${name}"
else
    python3 -m zipfile -c "${name}.zip" "${name}"
fi

n_wav="$(find "${dest}" -name '*.wav' | wc -l | tr -d ' ')"
size="$(du -h "${name}.zip" | cut -f1)"
echo "${dest}/ (${n_wav} wav files) -> ${repo_root}/${name}.zip (${size})"
