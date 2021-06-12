@echo off

:start 
cls

conda create -n %CD%/python3.6 python=3.6 anaconda
source activate %CD%/python3.6

pip install -r requirements.txt

