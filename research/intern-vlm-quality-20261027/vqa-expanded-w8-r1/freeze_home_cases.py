from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent
assert not (P/'controller-status.json').exists()
for name in ['vqa-cases.json','frozen-cases.sha256','selection-en.json']:
 f=P/name; backup=P/('scope1-superseded-'+name)
 if f.exists() and not backup.exists():shutil.copy2(f,backup)
x=json.loads((P/'scope2-candidates.json').read_text());rows=[]
def add(group,i,category,q,ref,custom=False):
 c=x[group][i].copy();old=c['id'];cid=f'home{len(rows)+1:03d}'
 c.update(id=cid,image=cid,source_question_id=old,category=category,question=q,reference_zh=ref,label_provenance='manual visible-fact annotation before inference' if custom else 'Chinese translation of VQAv2 consensus',index=len(rows)+1)
 shutil.copy2(P/'images'/(old+'.jpg'),P/'images'/(cid+'.jpg'))
 c['image_sha256']=hashlib.sha256((P/'images'/(cid+'.jpg')).read_bytes()).hexdigest();rows.append(c)
for i,q,ref in [(0,'新娘的头发是什么颜色？','棕色'),(1,'女士的头发是什么颜色？','金色'),(3,'孩子的头发是什么颜色？','金色'),(4,'打电话的女士头发是什么颜色？','黑色'),(6,'趴在电脑旁的男孩头发是什么颜色？','金色'),(8,'拿香蕉的孩子头发是什么颜色？','金色'),(10,'女孩的头发是什么颜色？','金色'),(11,'女士的头发是什么颜色？','红色或红棕色'),(12,'男孩的头发是什么颜色？','棕色'),(14,'孩子的头发是什么颜色？','棕色')]:add('hair',i,'hair',q,ref)
for i,q,ref,custom in [(1,'女士的可见表情更像微笑还是皱眉？','皱眉或严肃，没有微笑',True),(3,'小女孩在微笑吗？','是',False),(4,'孩子们在微笑吗？','没有',False),(5,'使用电脑的男子在微笑吗？','没有',False),(7,'坐在床上的女孩在微笑吗？','是',True),(8,'马旁边的女士在微笑吗？','是',False),(9,'戴帽子的孩子在微笑吗？','是',False),(11,'拿着手机的男子在微笑吗？','没有',False),(16,'坐在沙发上的男子在微笑吗？','没有，表情严肃',True),(17,'拿披萨的几个人都在微笑吗？','是',False)]:add('expression',i,'expression',q,ref,custom)
for i,q,ref,custom in [(0,'桌上有几台电脑？','两台',False),(1,'盘子上的黑色餐具是什么？','叉子',True),(3,'电脑屏幕下方桌上的红色物品是什么？','鼠标',False),(5,'盘子里主要是什么食物？','巧克力蛋糕',True),(6,'桌上有没有长条面包？','有',False),(8,'盘子中间靠后的几个深色容器是什么？','瓶子或啤酒瓶',True),(10,'桌上是什么水果？','苹果',False),(12,'桌上有几盏台灯？','一盏',False),(16,'桌上用于剪东西的工具是什么？','剪刀',False),(17,'桌上有几个音箱？','两个',False)]:add('desk',i,'desktop_object',q,ref,custom)
for i,q,ref in [(0,'按画面左右看，笔记本电脑在台式显示器哪一侧？','右侧'),(1,'黑色叉子是在盘子里还是盘子外？','盘子里或盘子上'),(3,'按画面左右看，笔记本电脑在大显示器哪一侧？','右侧'),(5,'蛋糕是放在盘子上还是直接放在桌上？','盘子上'),(6,'按画面左右看，长条面包在主要酒瓶的哪一侧？','左侧'),(8,'按画面上下看，深色瓶子在餐盘的上方还是下方？','上方'),(10,'苹果是在笔记本电脑前面还是后面？','前面'),(12,'按画面左右看，白色台灯在显示器哪一侧？','左侧'),(16,'按画面左右看，蓝色剪刀在彩色织物哪一侧？','左侧'),(17,'按画面上下看，键盘在显示器上方还是下方？','下方')]:add('desk',i,'desktop_relation',q,ref,True)
for i,ref in [(0,'厨房'),(2,'厨房'),(3,'厨房'),(6,'浴室或卫生间'),(7,'卧室或酒店卧室'),(8,'浴室或卫生间'),(11,'浴室或卫生间'),(13,'客厅'),(14,'卧室'),(15,'客厅')]:add('indoor',i,'indoor','这主要是什么房间？',ref)
for i,q,ref in [(0,'这主要是什么室外场景？','海滩或海边'),(1,'这主要是什么室外场景？','海滩或海边'),(2,'人们所在的主要场所是什么？','公园'),(3,'拿着冲浪板的人在哪里？','海滩或海边'),(4,'人们所在的主要场所是什么？','公园或公园草地'),(6,'人们所在的主要场所是什么？','公园或公园草地'),(7,'羊站在什么地方？','道路'),(8,'马在什么地方？','街道'),(9,'交通锥放在什么地方？','人行道'),(10,'摩托车所在的位置是什么地方？','人行道')]:add('outdoor',i,'outdoor',q,ref)
assert len(rows)==60
obj={'system':'你是视觉问答助手。请仅根据当前图片，用简短中文回答用户的问题；不要猜测看不到的细节，不确定时明确说明。','dataset':'Home companion focused development set: public COCO/VQAv2 images, 60 Chinese questions, 50 unique images; source consensus plus explicitly marked manual labels. Not official benchmark accuracy.','selection':'Frozen before any expanded board inference; no answer-based selection; original broad 60 superseded without inference. Full-frame resize, no face crops. Hair labels skew blonde; limited expression diversity.','scoring':'Manual core-fact correctness, accept equivalent synonyms; contradictions, unrelated lists, wrong reference objects fail; uncertainty separately tagged but counts incorrect. Expression means visible expression only, not internal emotion. Source annotations retained as provenance; manual questions do NOT inherit source consensus as their ground truth.','cases':rows}
obj['unique_images']=len(set(c['image_id'] for c in rows))
(P/'vqa-cases.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
(P/'frozen-cases.sha256').write_text(hashlib.sha256((P/'vqa-cases.json').read_bytes()).hexdigest()+'  vqa-cases.json\n')
print(len(rows),obj['unique_images'],(P/'frozen-cases.sha256').read_text())
