@echo off

setlocal

chdir ..\database

del statistics_db
sqlite3.exe statistics_db "create table statistics (timestamp varchar(100),config_name varchar(100),expansion_kernel_blocks int,expansion_kernel_threads int,collision_detection_kernel_blocks int,collision_detection_kernel_threads int,primary_roadnetwork_expansion_time int,collisions_computation_time float,primitives_extraction_time float,secondary_roadnetwork_expansion_time float,memory_copy_gpu_cpu_time float,memory_copy_cpu_gpu_time float,num_vertices int,num_edges int,num_collisions int,memory_in_use int);"

endlocal