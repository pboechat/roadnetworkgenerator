@echo off

setlocal ENABLEDELAYEDEXPANSION

rem ======================
rem	SPATIAL DIVISION TESTS
rem ======================

set /a c=0
set /a t=6*%REPETITIONS%*2

for /l %%i in (1,1,6) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo *****************************
		echo SPATIAL DIVISION TEST !c!/%t%
		echo *****************************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\botafogo_bay_sd_%%i_gpu.config true false %RUNS_DIR%
	)
)

for /l %%i in (1,1,6) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo *****************************
		echo SPATIAL DIVISION TEST !c!/%t%
		echo *****************************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_sd_%%i_gpu.config true false %RUNS_DIR%
	)
)

endlocal
