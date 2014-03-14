@echo off

setlocal ENABLEDELAYEDEXPANSION

:stringloop
if "!line!" EQU "" goto end
	for /f "delims=|" %%a in ("!line!") do set substring=%%a
	echo|set /p=!substring!; >> !outputFile!
:striploop
    set stripchar=!line:~0,1!
    set line=!line:~1!
    if "!line!" EQU "" goto stringloop
    if "!stripchar!" NEQ "|" goto striploop
    goto stringloop
:end

endlocal