# DataAssimilation

This is a sample dispatching program for data assimilation using 3-dimensional variation method.

Key modules include: 
WRF(weather forecast model), WRFvar(DA model), obsproc(observation pre-process), 
update_bc(boundary condition updating), EOFQC(quality control using Empirical Orthogonal Function)

Main procedures include: 
checking data availability, data quality control; writing control files (namelist) and executing each module;
I/O among modules.

## DA Scheme flowchart

![featpnt](https://github.com/wangminzheng/DataAssimilation/blob/master/pic/DA_flowchart.png)

