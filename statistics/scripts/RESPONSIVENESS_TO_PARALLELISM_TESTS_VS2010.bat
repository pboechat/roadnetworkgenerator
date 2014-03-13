echo off

setlocal

call SETENV.bat

chdir ..\..\build\mak.vc10\x32\src

mkdir %RUNS_DIR%

set REPETITIONS=10

rem ================================
rem	PARALLELISM TESTS (botafogo bay)
rem ================================

for /l %%i in (1,1,12) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\botafogo_bay_pr_%%i_gpu.config true true %RUNS_DIR%
	)
)

rem =======================================
rem	PARALLELISM TESTS (rio de janeiro city)
rem =======================================

for /l %%i in (1,1,12) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_pr_%%i_gpu.config true true %RUNS_DIR%
	)
)

endlocal
