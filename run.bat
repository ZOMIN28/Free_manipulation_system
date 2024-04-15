@echo off
rem Activate conda environment
call conda activate pytorch

python main.py

rem Close conda environment
call conda deactivate
