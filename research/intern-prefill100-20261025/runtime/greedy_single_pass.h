/* Optional exact first-maximum argmax. A single mapped-buffer read pass.
 * GCC vector extensions share identical source on x86 host and ARM64 NEON.
 * 16-bit indices cover current per-shard counts<=49152. Larger vectors use
 * the untouched baseline; no wrapping or cross-shard raw-score comparison. */
#ifndef OPI_GREEDY_SINGLE_PASS_H
#define OPI_GREEDY_SINGLE_PASS_H
#include <stdint.h>
#include <string.h>
#include <limits.h>
typedef int16_t opi_i16x8 __attribute__((vector_size(16)));
typedef uint16_t opi_u16x8 __attribute__((vector_size(16)));
static unsigned greedy_i16_single_pass(const int16_t *values,unsigned count) {
    if(count>UINT16_MAX)return greedy_i16(values,count);
    unsigned i=0,best=0;int16_t maximum=INT16_MIN;
    if(count>=8){
        opi_i16x8 largest={INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN};
        opi_u16x8 positions={0,1,2,3,4,5,6,7},indices=positions,step={8,8,8,8,8,8,8,8};
        for(;count-i>=8;i+=8){
            opi_i16x8 x;memcpy(&x,values+i,sizeof(x));
            opi_i16x8 improved=x>largest;
            largest=(improved&x)|(~improved&largest);
            opi_u16x8 mask=(opi_u16x8)improved;
            indices=(mask&positions)|(~mask&indices);
            positions+=step;
        }
        best=UINT_MAX;
        for(unsigned lane=0;lane<8;lane++){
            int16_t value=largest[lane];unsigned index=indices[lane];
            if(value>maximum||(value==maximum&&index<best)){maximum=value;best=index;}
        }
    }
    for(;i<count;i++)if(values[i]>maximum){maximum=values[i];best=i;}
    return best;
}
#endif
