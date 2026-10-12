#include "greedy_i16.h"
#include <stdio.h>
int main(void) {
    int16_t values[151674];uint32_t state=123;
    const unsigned lengths[]={1,7,8,9,17,257,49152,151674};
    for(unsigned l=0;l<sizeof(lengths)/sizeof(lengths[0]);l++){
        unsigned n=lengths[l];
        for(unsigned trial=0;trial<20;trial++){
            for(unsigned i=0;i<n;i++){state=1664525*state+1013904223;values[i]=(int16_t)(state>>16);}
            if(trial==0)for(unsigned i=0;i<n;i++)values[i]=INT16_MIN;
            if(trial==1){for(unsigned i=0;i<n;i++)values[i]=-31;values[n-1]=-30;}
            if(trial==2){values[n/2]=INT16_MAX;values[n-1]=INT16_MAX;}
            unsigned expected=0;
            for(unsigned i=1;i<n;i++)if(values[i]>values[expected])expected=i;
            if(greedy_i16(values,n)!=expected)return 1;
        }
    }
    puts("PASS: scalar/NEON greedy, negative values, tails and first-index ties");return 0;
}
