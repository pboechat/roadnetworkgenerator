echo off

set EXEC_BIN=Release\roadnetworkgenerator.exe
set CONFIGS_DIR=..\..\..\..\statistics\configs
set RUNS_DIR=..\..\..\..\statistics\runs\%date:~-4%_%date:~3,2%_%date:~0,2%_%time:~0,2%_%time:~3,2%_%time:~6,2%
set REPORT_FILE=%RUNS_DIR%\report.txt