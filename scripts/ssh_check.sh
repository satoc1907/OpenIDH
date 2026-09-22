#!/bin/bash
# Which HPC hosts can this machine reach over ssh, and with which key?
#
#   bash scripts/ssh_check.sh                 # uses ~/.ssh/config aliases below
#   bash scripts/ssh_check.sh raiden rikyu    # or name the aliases/hosts to try
#
# Prints, per host: whether the name resolves in ~/.ssh/config, whether a key
# is loaded, and the result of a 10-second non-interactive login. Nothing is
# changed; no password prompt is ever shown (BatchMode=yes) — a host that needs
# a password or an unregistered key shows as "auth failed", which is the answer.
HOSTS=("$@")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=(raiden rikyu)

echo "== ~/.ssh contents (public keys only) =="
ls -la ~/.ssh 2>/dev/null | grep -v "^total" || echo "  (no ~/.ssh)"
echo
echo "== Host entries in ~/.ssh/config =="
grep -iE "^\s*Host\s" ~/.ssh/config 2>/dev/null || echo "  (no ~/.ssh/config)"
echo
echo "== keys in ssh-agent =="
ssh-add -l 2>/dev/null || echo "  (agent not running or no keys loaded)"
echo

for h in "${HOSTS[@]}"; do
  echo "== $h =="
  # what ssh would actually use for this name
  ssh -G "$h" 2>/dev/null | grep -E "^(hostname|user|port|identityfile) " | sed 's/^/  /'
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$h" \
        'echo OK host=$(hostname) user=$(whoami) home=$HOME; ls -d OpenIDH 2>/dev/null' 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "  LOGIN OK: $out" | sed 's/^/  /'
  elif echo "$out" | grep -qi "permission denied\|publickey"; then
    echo "  auth failed (key not registered on $h, or wrong user): $out" | head -2
  elif echo "$out" | grep -qi "could not resolve\|timed out\|no route\|connection refused"; then
    echo "  unreachable (network / hostname): $out" | head -2
  else
    echo "  failed (rc=$rc): $out" | head -3
  fi
  echo
done
