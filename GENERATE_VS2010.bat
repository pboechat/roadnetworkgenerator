@echo off

mkdir build
chdir build

mkdir mak.vc10
chdir mak.vc10

mkdir x32
chdir x32

REM del CMakeCache.txt

cmake -G "Visual Studio 10" ../../../

if %errorlevel% NEQ 0 goto error
goto end

:error
echo Houve um erro. Pressione qualquer tecla para finalizar.
pause >nul

:end

REM pause >nul

cd ../../../