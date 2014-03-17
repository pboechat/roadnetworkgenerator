@echo off

setlocal ENABLEDELAYEDEXPANSION

rem =================
rem	PARALLELISM TESTS
rem =================

set /a c=0
set /a t=12*%REPETITIONS%*2

for /l %%i in (1,1,12) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		echo *************************
		echo PARALLELISM  TEST !c!/%t%
		echo *************************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\botafogo_bay_pr_%%i_gpu.config true false %RUNS_DIR%
	)
)

for /l %%i in (1,1,12) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		echo *************************
		echo PARALLELISM  TEST !c!/%t%
		echo *************************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_pr_%%i_gpu.config true false %RUNS_DIR%
	)
)

endlocal
