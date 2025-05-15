@echo off
REM Set OpenMP environment variable to fix the duplicate library issue
set KMP_DUPLICATE_LIB_OK=TRUE
echo OpenMP environment variable set: KMP_DUPLICATE_LIB_OK=TRUE
echo.
echo You can now run your Python scripts or launch Jupyter Notebook
echo without encountering the OpenMP duplicate library error.
echo.
cmd /k
