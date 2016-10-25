fd: fd.cu fd.hh makefile
	nvcc -O0 -arch=sm_35 -lpthread -lreadline -o fd fd.cu
rem:
	@rm -rf *.npy
