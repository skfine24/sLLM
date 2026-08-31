# env_source.sh -- source config.env WITHOUT clobbering real environment
# variables (plain `. config.env` would overwrite them; the documented
# precedence is env var > config.env > code default). Also tolerates CRLF.
#
# usage:  . /path/to/env_source.sh
#         sllm_load_env /path/to/config.env
sllm_load_env() {
  _sllm_cfg="$1"
  [ -f "$_sllm_cfg" ] || return 0
  while IFS= read -r _sllm_line || [ -n "$_sllm_line" ]; do
    _sllm_line="${_sllm_line%$'\r'}"
    case "$_sllm_line" in
      '' | ' '* | '#'*) continue ;;
    esac
    case "$_sllm_line" in
      *=*) ;;
      *) continue ;;
    esac
    _sllm_k="${_sllm_line%%=*}"
    _sllm_v="${_sllm_line#*=}"
    case "$_sllm_k" in
      [A-Za-z_]*) ;;
      *) continue ;;
    esac
    case "$_sllm_k" in
      *[!A-Za-z0-9_]*) continue ;;
    esac
    # only set when currently unset or empty: real env vars win
    if [ -z "$(eval "echo \"\${$_sllm_k:-}\"")" ]; then
      export "$_sllm_k=$_sllm_v"
    fi
  done < "$_sllm_cfg"
  unset _sllm_cfg _sllm_line _sllm_k _sllm_v
}
