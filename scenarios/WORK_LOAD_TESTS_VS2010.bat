rem @echo off

chdir ..\build\mak.vc10\x32\src

Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/botafogo_bay_ld_1_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/botafogo_bay_ld_2_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/botafogo_bay_ld_3_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/botafogo_bay_ld_4_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/botafogo_bay_ld_5_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/rio_de_janeiro_city_ld_1_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/rio_de_janeiro_city_ld_2_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/rio_de_janeiro_city_ld_3_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/rio_de_janeiro_city_ld_4_gpu.config true true ../../../../scenarios/dump
Release\roadnetworkgenerator.exe 1024 768 ../../../../scenarios/rio_de_janeiro_city_ld_5_gpu.config true true ../../../../scenarios/dump
