static inline int llama_is_eos(int token,int configured){return token==configured||token==128001||token==128008||token==128009;}
