fd: fd.cu fd.hh makefile
	nvcc -O0 -arch=sm_20 -lpthread -lreadline -o fd fd.cu
