rem ====================================================
rem REQUIRES STARTTIME AND ENDTIME ENVIRONMENT VARIABLES
rem ====================================================
rem SOURCE: http://stackoverflow.com/questions/9922498/calculate-time-difference-in-batch-file

rem convert STARTTIME and ENDTIME to milliseconds
set /A STARTTIME=(1%STARTTIME:~0,2%-100)*3600000 + (1%STARTTIME:~3,2%-100)*60000 + (1%STARTTIME:~6,2%-100)*1000 + (1%STARTTIME:~9,2%-100)*10
set /A ENDTIME=(1%ENDTIME:~0,2%-100)*3600000 + (1%ENDTIME:~3,2%-100)*60000 + (1%ENDTIME:~6,2%-100)*1000 + (1%ENDTIME:~9,2%-100)*10

set /A DURATION=%ENDTIME%-%STARTTIME%

if %ENDTIME% LSS %STARTTIME% set set /A DURATION=%STARTTIME%-%ENDTIME%

rem now break the milliseconds down to hours, minutes, seconds and the remaining milliseconds
set /A DURATIONH=%DURATION% / 3600000
set /A DURATIONM=(%DURATION% - %DURATIONH%*360000) / 60000
set /A DURATIONS=(%DURATION% - %DURATIONH%*360000 - %DURATIONM%*6000) / 1000
set /A DURATIONHS=(%DURATION% - %DURATIONH%*360000 - %DURATIONM%*6000 - %DURATIONS%*100) / 10

if %DURATIONH% LSS 10 set DURATIONH=0%DURATIONH%
if %DURATIONM% LSS 10 set DURATIONM=0%DURATIONM%
if %DURATIONS% LSS 10 set DURATIONS=0%DURATIONS%
if %DURATIONHS% LSS 10 set DURATIONHS=0%DURATIONHS%
