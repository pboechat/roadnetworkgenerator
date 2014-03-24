@echo off

setlocal

chdir ..\..\build\mak.vc10\x32\src

set EXEC_BIN=Release\roadnetworkgenerator.exe
set CONFIGS_DIR=..\..\..\..\presentation\configs
set SCRIPTS_DIR=..\..\..\..\presentation\scripts
set FRAMES_DIR=..\..\..\..\presentations\frames

set /a c=0

for /l %%i in (1,1,#NUM_CONFIGS#) do (
	set /a c=!c!+1
	echo **********************
	echo CONFIG. !c!/#NUM_CONFIGS#
	echo **********************
	start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\#BASE_CONFIG_FILE#_%%i.config false true %FRAMES_DIR%
)

endlocal
