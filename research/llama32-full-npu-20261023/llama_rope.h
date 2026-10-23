#ifndef LLAMA_SCALED_ROPE_H
#define LLAMA_SCALED_ROPE_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static float llama_inv[MODEL_HD/2];
static void llama_load_rope(const char *path){
 FILE *f=fopen(path,"rb");if(!f){perror(path);exit(2);}
 if(fread(llama_inv,sizeof(float),MODEL_HD/2,f)!=MODEL_HD/2||fgetc(f)!=EOF){fprintf(stderr,"bad scaled rope table\n");exit(2);}fclose(f);
 for(int i=0;i<MODEL_HD/2;i++)if(!isfinite(llama_inv[i])||llama_inv[i]<=0){fprintf(stderr,"invalid inverse frequency\n");exit(2);}
}
static inline float llama_rope_angle(int pos,int j){return pos*llama_inv[j%(MODEL_HD/2)];}
#include "llama_eos.h"
#endif
