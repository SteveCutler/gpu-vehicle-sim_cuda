# Serial Implementation for later CUDA Porting

TO DO:

-color code goals and vehicles together
-shrink vehicle size
-implement some measurment of error mechanism to compare GPU version against
-create a CPU benchmark
-introduce more comprehensive and robust error handling with cleanup

// 
move loop inside kernel to batch process timesteps
    -pingpong buffer?
implement environmental disturbances in parallel fashion
one vehicle per thread vs multi vehicle per thread

add boundary conditions for movement so vehicles bounce on boundary collision?
implement position sampling from float values for disturbance reading