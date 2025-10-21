@echo off
echo ========================================
echo   Pushing to GitHub
echo ========================================
echo.
echo Please replace YOUR_GITHUB_USERNAME with your actual GitHub username
echo.
pause

REM Replace YOUR_GITHUB_USERNAME with your actual username
git branch -M main
git remote add origin https://github.com/KhuzaimaHassan/AI-Academic-Mentor.git
git push -u origin main

echo.
echo ========================================
echo   Push Complete!
echo ========================================
pause

