@echo off

setlocal

chdir ..\..\build\mak.vc10\x32\src

call ..\..\..\..\statistics\scripts\SETENV.bat

mkdir %RUNS_DIR%

set REPETITIONS=2

set STARTTIME=%time%

rem ============================
rem	SPATIAL TESTS (botafogo bay)
rem ============================

for /l %%i in (1,1,6) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%/botafogo_bay_sd_%%i_gpu.config true true %RUNS_DIR%
	)
)


rem ===================================
rem	SPATIAL TESTS (rio de janeiro city)
rem ===================================

for /l %%i in (1,1,6) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_sd_%%i_gpu.config true true %RUNS_DIR%
	)
)

set ENDTIME=%time%

call ..\..\..\..\statistics\scripts\TIMEDIFF.bat

type %DURATION% > Elapsed time: %REPORT_FILE% (ms)

endlocal
