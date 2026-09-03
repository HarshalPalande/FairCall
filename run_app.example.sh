#!/usr/bin/env bash
# Template for running the Streamlit app against Razorpay Test Mode.
#
#   cp run_app.example.sh run_app.sh
#   # paste your own Test Mode keys into run_app.sh
#   ./run_app.sh
#
# run_app.sh is gitignored precisely because it holds a secret. This template
# is the only version that belongs in version control.
#
# Without these variables the app still runs — app/pages/3_Razorpay_Integration.py
# falls back to four built-in sample payments and says so on screen. The keys only
# enable browsing real Test Mode payments from your own account.
set -euo pipefail

export RAZORPAY_KEY_ID=rzp_test_replace_me
export RAZORPAY_KEY_SECRET=replace_me

# If your venv is not the in-repo .venv, make has to be told where it is -
# otherwise this dies at venv-check. Export VENV before running:
#
#     export VENV=$HOME/.venvs/ai-risk-manager
#
# or pass it through here as a default:  make app VENV="${VENV:-...}"
make app
