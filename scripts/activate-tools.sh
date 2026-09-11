# Source this from a shell: source scripts/activate-tools.sh
drugforge_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$drugforge_root/.tools/bin:$PATH"
export P2RANK_PATH="$drugforge_root/.tools/p2rank_2.5.1"
export JAVA_HOME="$drugforge_root/.tools/java25"
export HF_HOME="$drugforge_root/.tools/huggingface"
unset drugforge_root
