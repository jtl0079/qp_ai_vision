

@echo off

set conda_env_name=rsw_ai
set conda_env_config_file_name=rsw_ai_conda_env.yml

echo ==================================
echo Creating Conda Environment...
echo ==================================


call conda env list | findstr "%conda_env_name%" >nul

if %errorlevel%==0 (
    echo Environment %conda_env_name% already exists.
) else (
    echo Creating environment...
    call conda env create -f %conda_env_config_file_name%

    if errorlevel 1 (
        echo.
        echo Environment creation failed.
        pause
        exit /b 1
    )
)

echo.
echo ==================================
echo Activating Environment...
echo ==================================

call conda activate %conda_env_name%

if errorlevel 1 (
    echo Failed to activate environment.
    pause
    exit /b 1
)

echo.
echo ==================================
echo Registering Jupyter Kernel...
echo ==================================

python -m ipykernel install --user ^
    --name %conda_env_name% ^
    --display-name "Python (%conda_env_name%)"


echo.
echo ==================================
echo Installing Project...
echo ==================================

pip install -e .

if errorlevel 1 (
    echo Failed to install project.
    pause
    exit /b 1
)
    

echo.
echo ==================================
echo Finished
echo ==================================

python --version

conda env list

pause