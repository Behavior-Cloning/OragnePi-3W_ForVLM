def schedule(static, dynamic, batch):
 pos=static;total=static+dynamic+8;compact=main=tail=scalar=physical=padded=tail_rows=0
 while pos<total:
  left=total-pos
  if pos<=128 and left>=64:
   n=64;compact+=1;physical+=64
  elif left>=batch:
   n=batch;main+=1;physical+=batch
  elif left==8:
   n=8;tail+=1;physical+=8;tail_rows=8
  elif left>=8 and pos+batch<=448:
   n=left;main+=1;physical+=batch;padded+=batch-left;tail_rows=left
  else:
   n=1;scalar+=1;physical+=1
  pos+=n
 return dict(compact_calls=compact,main_calls=main,true_tail8_calls=tail,scalar_prompt_tokens=scalar,batch_calls=compact+main+tail,executed_query_rows=physical,padded_query_rows=padded,tail_rows=tail_rows)
