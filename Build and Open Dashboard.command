#!/bin/bash
# Double-click me: rebuilds medcare_dashboard.html from the latest backend outputs and opens it.
cd "$(dirname "$0")" || exit 1
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
python3 html_frontend/build_html.py && open medcare_dashboard.html
