@echo off

setlocal

chdir ..\..\build\mak.vc10\x32\src

call %SCRIPTS_DIR%\SETENV.bat

mkdir %RUNS_DIR%

set REPETITIONS=5

set STARTTIME=%time%

call %SCRIPTS_DIR%\WORK_LOAD_TESTS_VS2010.bat
call %SCRIPTS_DIR%\SPATIAL_DIVISION_TESTS_VS2010.bat
call %SCRIPTS_DIR%\PARALLELISM_TESTS_VS2010.bat

set ENDTIME=%time%

call %SCRIPTS_DIR%\TIMEDIFF.bat

echo Elapsed time: %DURATION% (ms) > %REPORT_FILE%

endlocal
