@echo off
title Loan Default Prediction - Frontend Server
echo =====================================================================
echo  Starting Frontend Web Server...
echo  Serving at: http://localhost:3000
echo =====================================================================
python -m http.server 3000
pause

