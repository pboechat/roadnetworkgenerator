@echo off

setlocal ENABLEDELAYEDEXPANSION

rem ===============
rem	WORK LOAD TESTS
rem ===============

set /a c=0
set /a t=5*%REPETITIONS%*4

for /l %%i in (1,1,5) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo **********************
		echo WORK LOAD TEST !c!/%t%
		echo **********************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\botafogo_bay_pw_%%i_gpu.config true false %RUNS_DIR%
	)
)

for /l %%i in (1,1,5) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo **********************		
		echo WORK LOAD TEST !c!/%t%
		echo **********************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\botafogo_bay_sw_%%i_gpu.config true false %RUNS_DIR%
	)
)

for /l %%i in (1,1,5) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo **********************
		echo WORK LOAD TEST !c!/%t%
		echo **********************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_pw_%%i_gpu.config true false %RUNS_DIR%
	)
)

for /l %%i in (1,1,5) do (
	for /l %%j in (1,1,%REPETITIONS%) do (
		set /a c=!c!+1
		echo **********************
		echo WORK LOAD TEST !c!/%t%
		echo **********************
		start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\rio_de_janeiro_city_sw_%%i_gpu.config true false %RUNS_DIR%
	)
)

endlocal