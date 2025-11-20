#!/usr/bin/env bash
# BibTeX Reference Injector for OSR Methods
# Vote-Tracing Paper: Ensures all OSR citations are available

set -euo pipefail
BIB="${1:-refs.bib}"

ensure() {
    local key="$1"
    local stub="$2"
    if ! grep -q "@.*{$key" "$BIB"; then
        printf "\n%s\n" "$stub" >> "$BIB"
        echo "➕ added $key to $BIB"
    fi
}

# OSR method references
ensure "liu2020energy" "$(cat <<'EOF'
@inproceedings{liu2020energy,
  title={Energy-based Out-of-distribution Detection},
  author={Liu, Weitang and Wang, Xiaodong and Owens, John and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
EOF
)"

ensure "liang2018odin" "$(cat <<'EOF'
@inproceedings{liang2018odin,
  title={Enhancing the reliability of out-of-distribution image detection in neural networks},
  author={Liang, Shiyu and Li, Yixuan and Srikant, R},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
EOF
)"

ensure "lee2018simple" "$(cat <<'EOF'
@inproceedings{lee2018simple,
  title={A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks},
  author={Lee, Kimin and Lee, Kibok and Lee, Honglak and Shin, Jinwoo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
EOF
)"

ensure "mahalanobis1936" "$(cat <<'EOF'
@article{mahalanobis1936,
  title={On the generalised distance in statistics},
  author={Mahalanobis, Prasanta Chandra},
  journal={Proceedings of the National Institute of Sciences of India},
  volume={2},
  number={1},
  pages={49--55},
  year={1936}
}
EOF
)"

ensure "huang2021mos" "$(cat <<'EOF'
@inproceedings{huang2021mos,
  title={MOS: Towards Scaling Out-of-Distribution Detection for Large Semantic Space},
  author={Huang, Rui and Geng, Andrew and Li, Yixuan},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
EOF
)"

echo "✅ OSR bibliography entries ensured in $BIB"