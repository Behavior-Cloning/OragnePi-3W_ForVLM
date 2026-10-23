import pathlib,json,sentencepiece
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent;s=sentencepiece.SentencePieceProcessor(model_file=str(P/'model/tokenizer.model'));h=AutoTokenizer.from_pretrained(P/'model',local_files_only=True)
texts=['What type of building is it?','What color is the church?','Describe the image briefly','请用中文描述这张图片。','<image>Describe this image.\n','a\tb','hello\nworld','今日天气不错。']
rows=[dict(text=t,spm=s.encode(t),hf=h.encode(t,add_special_tokens=False)) for t in texts]
for r in rows:r['equal']=r['spm']==r['hf']
(P/'tokenizer-encoding-parity.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print([(r['text'],r['equal']) for r in rows])
