# Phần dùng chung của bộ lệnh craneops. Được `source`, không chạy trực tiếp.
#
# Shell chứ không phải Python: các lệnh này phải chạy được trên **máy đích**, nơi chỉ có
# Docker — không venv, không `uv`, không phụ thuộc gì của dự án. Một CLI cần cài đặt trước
# khi cài được hệ thống là một CLI vô dụng đúng lúc cần nhất.

set -euo pipefail

# Gốc repo, suy từ vị trí script — nên gọi được từ thư mục bất kỳ.
CRANEOPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CRANEOPS_ROOT

if [ -t 1 ]; then
  readonly C_DIM=$'\033[2m' C_RED=$'\033[31m' C_BOLD=$'\033[1m' C_OFF=$'\033[0m'
else
  readonly C_DIM='' C_RED='' C_BOLD='' C_OFF=''
fi

die() {
  printf '%s✗%s %s\n' "$C_RED" "$C_OFF" "$*" >&2
  exit 1
}

note() { printf '%s→ %s%s\n' "$C_DIM" "$*" "$C_OFF"; }

# Kiểm file env tồn tại và nói rõ cách tạo. Thiếu env là lỗi hay gặp nhất khi cài mới, và
# thông báo mặc định của docker compose không nói phải làm gì.
require_env() {
  local file="$1" example="$2"
  [ -f "$CRANEOPS_ROOT/$file" ] || die "thiếu $file
   Tạo từ mẫu:  cp $example $file
   rồi điền các giá trị trong đó."
}

CRANEOPS_NETWORK=craneops
readonly CRANEOPS_NETWORK

# Tạo network dùng chung nếu chưa có.
#
# Hai compose khai nó là `external` chứ không cùng khai `name:`: khai trùng tên thì compose
# cảnh báo "exists but was not created for project ..." ở MỌI lệnh, và `down` của project
# này sẽ thử xoá network mà project kia đang dùng. External thì quyền sở hữu rõ ràng —
# nó thuộc về bộ lệnh, không thuộc về service nào.
ensure_network() {
  docker network inspect "$CRANEOPS_NETWORK" >/dev/null 2>&1 && return 0
  local err
  if ! err="$(docker network create "$CRANEOPS_NETWORK" 2>&1)"; then
    # Trên máy dùng chung, nguyên nhân hay gặp nhất là hết dải địa chỉ vì network của
    # NGƯỜI KHÁC — thứ mình không được phép dọn. Nói rõ để không ai đi xoá nhầm.
    case "$err" in
      *"address pools"*) die "hết dải địa chỉ Docker — không tạo được network $CRANEOPS_NETWORK.
   Máy đang có $(docker network ls -q | wc -l) network. Xem của mình là những cái nào:
       docker network ls --filter label=com.docker.compose.project
   rồi dọn network compose CŨ CỦA MÌNH:  docker network prune
   ⚠️ Đừng xoá network của người khác trên máy dùng chung." ;;
      *) die "không tạo được network $CRANEOPS_NETWORK: $err" ;;
    esac
  fi
  note "đã tạo network $CRANEOPS_NETWORK"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "không tìm thấy docker"
  docker info >/dev/null 2>&1 || die "docker không chạy được (thiếu quyền? thử: docker info)"
  # Network là external nên phải tồn tại trước khi compose chạy. Gắn ở đây để mọi lệnh
  # dùng Docker đều được lo, không phải nhớ gọi riêng ở từng chỗ.
  ensure_network
}

# Đọc một biến từ file env, trả về mặc định nếu không có.
env_value() {
  local file="$1" key="$2" fallback="${3:-}"
  local v
  v="$(grep -E "^${key}=" "$CRANEOPS_ROOT/$file" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
  printf '%s' "${v:-$fallback}"
}

# Tên lệnh của một script, mỗi dòng một cái. KHÔNG màu, KHÔNG căn lề: đây là bản cho MÁY
# đọc.
#
# ⚠️ Đừng để code khớp mẫu đọc đầu ra của `print_commands`. Bản đó có mã màu khi stdout là
# terminal, và mã màu dính liền tên lệnh (`ESC[1mstatus`) nên `\bstatus\b` KHÔNG khớp —
# lỗi chỉ hiện khi chạy trong terminal thật, còn qua pipe thì im lặng chạy đúng.
list_commands() {
  grep -E '^\s*#:' "$1" | sed -E 's/^\s*#:\s?//' | cut -d'|' -f1 | tr -d ' \t'
}

# Bảng lệnh cho NGƯỜI đọc. Cùng nguồn `#:` với list_commands, nên help không trôi khỏi
# danh sách lệnh thật.
print_commands() {
  local script="$1"
  grep -E '^\s*#:' "$script" | sed -E 's/^\s*#:\s?//' | while IFS='|' read -r cmd desc; do
    printf '  %s%-16s%s %s\n' "$C_BOLD" "$cmd" "$C_OFF" "$desc"
  done
}

# Từ chối lệnh không biết, kèm danh sách hợp lệ — gõ sai lệnh không được im lặng không làm gì.
unknown_command() {
  local script="$1" cmd="$2"
  printf '%s✗%s lệnh không hợp lệ: %s\n\n' "$C_RED" "$C_OFF" "${cmd:-<trống>}" >&2
  print_commands "$script" >&2
  exit 2
}
