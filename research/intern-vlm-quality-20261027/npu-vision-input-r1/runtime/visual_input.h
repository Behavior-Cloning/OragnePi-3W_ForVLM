#ifndef OPI_VISUAL_INPUT_H
#define OPI_VISUAL_INPUT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* Single-threaded resident request; no image KV is reusable across requests. */
static float *opi_visual_rows;
static int opi_visual_start, opi_visual_count;
static inline float opi_input_value(const float *table,int token,int pos,int j,int dim) {
    if(opi_visual_rows && pos>=opi_visual_start && pos-opi_visual_start<opi_visual_count)
        return opi_visual_rows[(size_t)(pos-opi_visual_start)*dim+j];
    return table[(size_t)token*dim+j];
}
static inline int opi_read_visual(FILE *f,uint32_t count,int n,int dim,int *reuse_cap) {
    uint32_t meta[2];
    if(count<1 || count>64 || fread(meta,sizeof(uint32_t),2,f)!=2) return 0;
    if(meta[0]>(uint32_t)n || count>(uint32_t)n-meta[0] || meta[1]>meta[0]) return 0;
    size_t elements=(size_t)count*dim;
    float *v=malloc(elements*sizeof(float));if(!v)return 0;
    if(fread(v,sizeof(float),elements,f)!=elements){free(v);return 0;}
    for(size_t i=0;i<elements;i++)if(!isfinite(v[i])){free(v);return 0;}
    opi_visual_rows=v;opi_visual_start=(int)meta[0];opi_visual_count=(int)count;
    *reuse_cap=(int)meta[1];return 1;
}
static inline void opi_clear_visual(void) {
    free(opi_visual_rows);opi_visual_rows=NULL;opi_visual_start=opi_visual_count=0;
}
#endif
