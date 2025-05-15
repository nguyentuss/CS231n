import os

# Set environment variable to allow duplicate OpenMP libraries
# This is not ideal but allows the code to run without crashing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Print confirmation message
print("OpenMP environment variable set: KMP_DUPLICATE_LIB_OK=TRUE")
print("This will allow the code to run despite multiple OpenMP libraries being loaded.")
print("Note: This is a workaround. For a permanent solution, consider using a conda environment")
print("      with consistent MKL/OpenMP dependencies.")
