@echo off

setlocal ENABLEDELAYEDEXPANSION

del /Q ..\database\*.csv
del /Q tmp.output

set /a c=0
for /f "tokens=*" %%q in (..\database\queries.sql) do (
	..\database\sqlite3.exe ..\database\statistics_db -header -csv -separator ; "%%q" > tmp.output
	set outputFile=..\database\query!c!.csv
	del /Q !outputFile!
	for /f "tokens=*" %%r in (tmp.output) do (
		rem set line=%%r
		rem call ECHO_STRING_TO_CSV_FILE.bat
		echo %%r >> !outputFile!
	)
	set /a c=c+1
)

del /Q tmp.output

endlocal

