#include <assert.h>
#include <stdio.h>
#include "schedule_prefill.h"
int main(void){unsigned requests=0,choices=0;int bs[]={0,32,64,128,256};
 for(int prefix=64;prefix<=128;prefix+=64)for(unsigned bi=0;bi<5;bi++)for(int tail=0;tail<2;tail++)for(int pad=0;pad<2;pad++)for(int body=0;body<=320;body++){
  int prompt=prefix+body+8;if(prompt+31>512)continue;
  /* Cold system-only partition must stay M16 even if a larger bank exists. */
  int pos=0;while(pos<prefix){PrefillChoice x=choose_prefill(pos,prefix,prefix,512,bs[bi],tail,8,pad);assert(x.kind==OPI_BASE16&&x.rows==16&&x.valid==16);pos+=x.valid;choices++;}
  int coverage[512]={0};unsigned newrows=0;pos=prefix;
  while(pos<prompt){PrefillChoice x=choose_prefill(pos,prompt,prefix,512,bs[bi],tail,8,pad);
   assert(x.valid>0&&x.valid<=x.rows&&pos+x.rows<=512&&pos+x.valid<=prompt);
   if(x.kind==OPI_BULK){assert(x.rows==bs[bi]);if(x.valid!=x.rows)assert(pad&&x.valid>16&&x.valid==prompt-pos);}
   if(x.kind==OPI_TAIL8)assert(tail&&x.rows==8&&prompt-pos==8);
   for(int j=0;j<x.valid;j++){coverage[pos+j]++;}
   newrows+=x.valid;pos+=x.valid;choices++;
  }
  assert(newrows==(unsigned)(body+8)&&pos==prompt);
  for(int j=0;j<512;j++){assert(coverage[j]==(j>=prefix&&j<prompt));}
  requests++;
 }
 printf("{\"requests\":%u,\"choices\":%u,\"logical_coverage_exact\":true,\"system_always_m16\":true,\"tail_capacity_safe\":true}\n",requests,choices);return 0;}
