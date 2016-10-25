#define TEST 0

#if TEST == 0
// #define GD_X 64
// #define GD_Y 512
// #define BD_X 64
// #define BD_Y 8
// #define WRITEEVERY 100
#define ITERS 100000
#define SAVE 1000
// #define BCOND NFX
// #define FTYPE FLOAT
// #define ANISOTROPY NO
#endif

char* genFileName(char*, char*, int);
void* workerThread(void*);
void* controlThread(void*);
void commandHandler(char*);
void cumulativeTimerStart(cudaEvent_t);
void cumulativeTimerStop(cudaEvent_t, cudaEvent_t, float*);

#define SLEN 100
