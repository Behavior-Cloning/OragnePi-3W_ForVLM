/* Temporary SIMD preconditioner; terminated and reaped before inference. */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <unistd.h>

static volatile sig_atomic_t running=1;
static volatile float sink[8];
static void stop(int sig){(void)sig;running=0;}
static void *heat(void *arg){
    int core=(int)(intptr_t)arg;
    cpu_set_t set;CPU_ZERO(&set);CPU_SET(core,&set);
    if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set))return NULL;
    float32x4_t a[16];
    for(int i=0;i<16;i++)a[i]=vdupq_n_f32((float)(i+1));
    const float32x4_t x=vdupq_n_f32(.999999f),b=vdupq_n_f32(.000001f);
    while(running){
        for(int k=0;k<10000;k++)for(int i=0;i<16;i++)a[i]=vfmaq_f32(b,a[i],x);
        sink[core]=vgetq_lane_f32(a[core],0);
    }
    return NULL;
}
int main(void){
    signal(SIGTERM,stop);signal(SIGINT,stop);signal(SIGALRM,stop);alarm(300);
    if(prctl(PR_SET_PDEATHSIG,SIGTERM))return 2;
    if(getppid()==1)return 2;
    pthread_t threads[8];int made=0;
    for(int i=0;i<8;i++){
        if(pthread_create(&threads[i],NULL,heat,(void*)(intptr_t)i)){running=0;break;}
        made++;
    }
    for(int i=0;i<made;i++)pthread_join(threads[i],NULL);
    return made==8?0:2;
}
