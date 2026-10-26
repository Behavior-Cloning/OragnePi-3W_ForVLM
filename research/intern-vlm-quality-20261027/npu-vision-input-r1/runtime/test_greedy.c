#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include "greedy_i16.h"
#include "greedy_single_pass.h"
static unsigned cases=0;static uint32_t rng=0x7cbd1234;
static uint32_t random32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static unsigned oracle(const int16_t *v,unsigned n){unsigned best=0;for(unsigned i=1;i<n;i++)if(v[i]>v[best])best=i;return best;}
static void check(const int16_t *v,unsigned n){unsigned a=oracle(v,n),b=greedy_i16_single_pass(v,n),c=greedy_i16(v,n);if(a!=b||a!=c){fprintf(stderr,"argmax mismatch n%u expected%u single%u baseline%u\n",n,a,b,c);exit(3);}cases++;}
int main(void){
 int16_t *v=malloc(65538*sizeof(int16_t));if(!v)return 2;
 int16_t alphabet[]={INT16_MIN,-1,0,INT16_MAX};
 for(unsigned n=1;n<=8;n++){unsigned total=1U<<(2*n);for(unsigned code=0;code<total;code++){unsigned z=code;for(unsigned i=0;i<n;i++){v[i]=alphabet[z&3];z>>=2;}check(v,n);}}
 for(int value=INT16_MIN;value<=INT16_MAX;value++){
  for(unsigned i=0;i<17;i++){v[i]=(int16_t)value;}check(v,17);
  v[0]=INT16_MIN;v[8]=INT16_MIN;check(v,17);
 }
 unsigned lengths[]={1,7,8,9,15,16,17,31,32,33,4218,49152,65534,65535,65536,65537};
 for(unsigned z=0;z<sizeof(lengths)/sizeof(lengths[0]);z++){
  unsigned n=lengths[z];for(unsigned i=0;i<n;i++)v[i]=INT16_MIN;check(v,n);
  for(unsigned pos=0;pos<n;pos+=pos<40?1:997){v[pos]=INT16_MAX;check(v,n);v[pos]=INT16_MIN;}
  for(unsigned i=0;i<n;i++){v[i]=(i%8==7)?0:-1;}check(v,n);
 }
 for(unsigned z=0;z<2000;z++){
  unsigned n=1+random32()%65537;for(unsigned i=0;i<n;i++)v[i]=(int16_t)random32();
  if(z%3==0){unsigned a=random32()%n,b=random32()%n;v[a]=v[b]=INT16_MAX;}
  check(v,n);
 }
 free(v);printf("{\"cases\":%u,\"mismatches\":0,\"all_int16_values_tested\":true,\"exhaustive_lengths1to8_alphabet4\":true,\"random_cases\":2000,\"max_count\":65537}\n",cases);return 0;
}
