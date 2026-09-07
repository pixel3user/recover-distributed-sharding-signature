#!/bin/bash
set -u
mkdir -p /logs/verifier
if python3 -I -m pytest /tests/test_outputs.py -rA; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
