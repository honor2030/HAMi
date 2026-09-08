#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
chart_dir="${repo_root}/charts/hami"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

registry='registry.test'
monitor_repository='monitor'
monitor_tag='legacy-tag'
monitor_digest='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
plugin_repository='plugin'
plugin_tag='plugin-tag'
scheduler_repository='scheduler'
scheduler_tag='scheduler-tag'

common_values=(
  --set-string "global.imageRegistry=${registry}"
  --set-string "devicePlugin.monitor.image.repository=${monitor_repository}"
  --set-string "devicePlugin.monitor.image.tag=${monitor_tag}"
  --set-string "devicePlugin.image.repository=${plugin_repository}"
  --set-string "devicePlugin.image.tag=${plugin_tag}"
  --set-string "scheduler.extender.image.repository=${scheduler_repository}"
  --set-string "scheduler.extender.image.tag=${scheduler_tag}"
)

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_line_count() {
  local file="$1"
  local expected="$2"
  local want="$3"
  local got
  got="$(grep -Fxc "${expected}" "${file}" || true)"
  [[ "${got}" == "${want}" ]] || fail "${file} has ${got} occurrences of ${expected}; want ${want}"
}

baseline="${tmp_dir}/tag.yaml"
digest_render="${tmp_dir}/digest.yaml"
diff_file="${tmp_dir}/render.diff"

helm template monitor-image "${chart_dir}" "${common_values[@]}" >"${baseline}"
helm template monitor-image "${chart_dir}" "${common_values[@]}" \
  --set-string "devicePlugin.monitor.image.digest=${monitor_digest}" >"${digest_render}"

monitor_tag_image="          image: ${registry}/${monitor_repository}:${monitor_tag}"
monitor_digest_image="          image: ${registry}/${monitor_repository}@${monitor_digest}"
plugin_image="          image: ${registry}/${plugin_repository}:${plugin_tag}"
scheduler_image="          image: ${registry}/${scheduler_repository}:${scheduler_tag}"

assert_line_count "${baseline}" "${monitor_tag_image}" 1
assert_line_count "${baseline}" "${plugin_image}" 1
assert_line_count "${baseline}" "${scheduler_image}" 1
assert_line_count "${digest_render}" "${monitor_tag_image}" 0
assert_line_count "${digest_render}" "${monitor_digest_image}" 1
assert_line_count "${digest_render}" "${plugin_image}" 1
assert_line_count "${digest_render}" "${scheduler_image}" 1

set +e
diff -u "${baseline}" "${digest_render}" >"${diff_file}"
diff_status=$?
set -e
[[ "${diff_status}" == 1 ]] || fail "digest render diff exit status was ${diff_status}; want 1"

actual_changes="$(awk '/^(---|\+\+\+|@@)/ { next } /^[+-]/ { print }' "${diff_file}")"
expected_changes="$(printf -- '-%s\n+%s' "${monitor_tag_image}" "${monitor_digest_image}")"
[[ "${actual_changes}" == "${expected_changes}" ]] || fail "digest render changed more than the monitor image"

printf 'PASS: monitor digest rendering changes only the monitor image; tag fallback, device-plugin, and scheduler images are unchanged.\n'
