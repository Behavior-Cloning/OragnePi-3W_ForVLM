"""Build the versioned household atomic-action design inventory; no hardware calls."""
import json, hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
ROWS=[]
def add(group, code, name, effect, kind, resources, caps, tags, pre=None):
    ROWS.append(dict(group=group, code=code, name=name, description=effect, kind=kind,
        resources=resources.split(','), capabilities=caps.split(','), tags=tags.split(','), extra_preconditions=pre or []))
def block(group, kind, resources, caps, data):
    for line in data.strip().splitlines():
        code,name,effect,tags=line.split('|')
        add(group,code,name,effect,kind,resources,caps,tags)
# Movement axes are physically different; speed/amplitude variants do not receive IDs.
block('头部','pose','head','head_pan',"""
head_left|向左转头|头部朝机器人自身左侧停留；改变头朝向，不移动底座，也不识别目标。|head,left,attention
head_right|向右转头|头部朝机器人自身右侧停留；改变头朝向，不移动底座，也不识别目标。|head,right,attention
""")
block('头部','pose','head','head_tilt',"""
head_up|抬头|头部向上仰，停在上仰姿态；眼睛图案不随之更换。|head,up,attention
head_down|低头|头部向下俯，停在下俯姿态；不同时弯腰。|head,down,quiet
""")
block('头部','pose','head','head_roll',"""
head_roll_left|向左歪头|头部向自身左肩方向侧倾，保持侧倾；不是向左转脸。|head,left,curiosity
head_roll_right|向右歪头|头部向自身右肩方向侧倾，保持侧倾；不是向右转脸。|head,right,curiosity
""")
add('头部','head_center','头部回正','头部回到已标定的正向中立姿态；只复位头部。','pose','head','head_pan,head_tilt,head_roll','head,neutral')
add('头部','head_nod','点头一次','头部完成一次俯仰往返，回到进入时姿态，用于肯定回应。','cycle','head','head_tilt','head,agreement')
add('头部','head_shake','摇头一次','头部完成一次左右摆动并回到进入时姿态，用于否定回应。','cycle','head','head_pan','head,disagreement')
block('眼睛','render','eyes_gaze','display_eyes',"""
gaze_center|眼睛看正前方|眼睛画面中的瞳孔居中；不转动机械头部，不启动目标跟踪。|eyes,front,attention
gaze_left|眼睛看左侧|瞳孔画在机器人自身左侧，眼睛外轮廓不变；不转头。|eyes,left,attention
gaze_right|眼睛看右侧|瞳孔画在机器人自身右侧，眼睛外轮廓不变；不转头。|eyes,right,attention
gaze_up|眼睛看上方|瞳孔画在眼睛区域上部，眼睛外轮廓不变；不抬头。|eyes,up,thinking
gaze_down|眼睛看下方|瞳孔画在眼睛区域下部，眼睛外轮廓不变；不低头。|eyes,down,quiet
""")
block('眼睛','cycle','eyes_lids','display_eyes',"""
blink|眨眼一次|双眼眼睑完成一次闭合再恢复；不附带头部或嘴部变化。|eyes,blink,natural
wink_left|左眼眨一下|只有左眼闭合一次再恢复，右眼维持原状态。|eyes,left,playful
wink_right|右眼眨一下|只有右眼闭合一次再恢复，左眼维持原状态。|eyes,right,playful
""")
block('眼睛','render','eyes_lids','display_eyes',"""
eyes_closed|双眼闭合|双眼显示完全闭合的平直眼睑，保持闭眼；不是一次眨眼。|eyes,closed,rest
eyes_droop|半垂眼睑|上眼睑覆盖瞳孔上半部，下眼睑保持原位，呈困倦眼型。|eyes,sleepy,quiet
eyes_squint|眯眼|上下眼睑同时向中间收拢，留下窄缝，呈审视眼型；不闭死。|eyes,squint,curiosity
eyes_wide|睁圆眼睛|上下眼睑张开，显示完整圆形瞳孔轮廓，呈惊讶眼型。|eyes,wide,surprise
""")
add('眼睛','eyes_spiral','圈圈眼','双眼区域以螺旋图案替换瞳孔和眼睑，表示晕乎或困惑；不是视觉检测。','render','eyes_gaze,eyes_lids','display_eyes','eyes,confused,cartoon')
block('嘴部','render','mouth','display_mouth',"""
mouth_neutral|嘴部恢复平静|嘴部显示一条水平短线，不改变眼睛、眉毛或脸颊。|mouth,neutral,quiet
mouth_smile|微笑嘴型|嘴角上扬，嘴唇闭合，显示一条微笑弧线。|mouth,smile,friendly
mouth_frown|难过嘴型|嘴角下垂，显示与微笑相反的弧线。|mouth,sad,comfort
mouth_o|惊讶圆嘴|嘴部显示圆形开口，表示吃惊；不输出声音。|mouth,surprise,round
mouth_pout|嘟嘴|双唇向中央聚拢，显示突出的双弧唇形；不发出亲吻声。|mouth,pout,playful
mouth_teeth|咬牙嘴型|嘴部显示上下相合的齿列矩形，表示用力或紧张；不是咀嚼。|mouth,tension,effort
mouth_tongue|吐舌表情|嘴部显示伸出的舌头图形；只有屏幕变化。|mouth,tongue,playful
mouth_wavy|为难波浪嘴|嘴线左右起伏成波浪形，表示犹豫或为难。|mouth,hesitant,wavy
mouth_zip|拉链嘴表情|嘴部区域显示闭合拉链图案，表示安静；不改变麦克风或音量。|mouth,quiet,zipper
""")
block('眉毛','render','brows','display_brows',"""
brows_neutral|眉毛恢复平直|两条眉毛在中立高度保持水平，不改变眼睑。|brows,neutral
brows_raise|双眉上扬|两条眉毛整体向上抬，保留水平走向，表示惊讶或期待。|brows,surprise,expectation
brows_knit|皱眉|双眉内端降低并靠近，形成向鼻梁下压的斜线，表示认真或不满。|brows,serious,disagreement
brows_worry|担忧眉|双眉内端抬高、外端降低，形成担忧的八字眉，与皱眉方向相反。|brows,worry,comfort
""")
block('脸颊','render','cheeks','display_cheeks',"""
cheeks_clear|脸颊恢复干净|移除脸颊区域装饰，恢复基础脸色；不清空其他面部区域。|cheeks,neutral
cheeks_blush|脸颊泛红|双颊显示成对红晕色块，表示害羞或亲近。|cheeks,shy,friendly
cheeks_sweat|脸颊汗滴|单侧脸颊显示一枚蓝色汗滴，表示尴尬或费劲。|cheeks,awkward,effort
cheeks_puff|鼓腮表情|脸颊两侧外轮廓呈鼓起圆弧，表示憋气或赌气；不改变嘴型。|cheeks,pout,cartoon
""")
block('上身','pose','torso','torso_pitch',"""
torso_forward|上身前倾|上身在原地向前倾，底座位置不变，用于靠近式倾听姿态。|torso,forward,listening
torso_back|上身后仰|上身在原地向后倾，底座位置不变，用于退让或惊讶姿态。|torso,back,surprise
""")
block('上身','pose','torso','torso_roll',"""
torso_left|上身向左侧倾|上身向自身左侧倾斜，不移动底座。|torso,left,playful
torso_right|上身向右侧倾|上身向自身右侧倾斜，不移动底座。|torso,right,playful
""")
block('上身','pose','torso','torso_yaw',"""
torso_turn_left|上身向左转|腰部使上身朝左转，底座和头部控制通道不主动变化。|torso,left,attention
torso_turn_right|上身向右转|腰部使上身朝右转，底座和头部控制通道不主动变化。|torso,right,attention
""")
add('上身','torso_center','上身回正','上身回到已标定直立中立姿态；不同时收手或回正头部。','pose','torso','torso_pitch,torso_roll,torso_yaw','torso,neutral')
add('上身','torso_sway','上身左右摇摆一次','上身完成一个小幅左右往返并回到进入姿态，不移动底座。','cycle','torso','torso_roll','torso,rhythm,playful')
add('上身','torso_bob','上身前后轻摆一次','上身完成一个小幅前后往返并回到进入姿态，不自动配合音乐节拍。','cycle','torso','torso_pitch','torso,rhythm,playful')
block('肩部','pose','shoulders','shoulder_elevation',"""
shoulder_left|抬左肩|只抬高左肩，手臂保持被动跟随，不附加挥手。|shoulder,left
shoulder_right|抬右肩|只抬高右肩，手臂保持被动跟随，不附加挥手。|shoulder,right
shoulders_relax|双肩放松|双肩回到已标定放松高度，不改变手指手势。|shoulder,neutral,relax
""")
add('肩部','shoulders_shrug','耸肩一次','双肩同步抬起后放回进入高度，表达不确定；不附带摊手。','cycle','shoulders','shoulder_elevation','shoulder,uncertain')

# Left/right are distinct actuators, not aliases. Each arm entry is one pose or one gesture.
arm_defs=[
('rest','手臂自然放下','手臂回到身体侧面的休息位置，不改变手指形状。','pose','rest'),
('raise','举手','手臂向上举至已标定举手姿态，表达回应或招呼；不挥动。','pose','greeting'),
('forward','手臂向前伸','手臂向前平伸，停在空中；不触碰或递取物品。','pose','forward'),
('side','手臂向侧面伸','手臂向自身同侧伸开，用于方向示意，不要求伸出食指。','pose','side'),
('chest','手收在胸前','前臂折向胸前并保持间隙，表现自我指示或礼貌回应；不拍胸。','pose','self'),
('hip','叉腰姿态','手停在同侧腰部外缘附近，保持自体间隙，形成叉腰轮廓；不压住身体。','pose','confident'),
('cheek','托腮姿态','手停在同侧脸颊下方附近，保持间隙，呈思考托腮轮廓；不承重接触。','pose','thinking'),
('salute','敬礼姿态','手停在同侧额前外方，形成敬礼轮廓；不触碰头部，不附加低头。','pose','respect'),
('overhead_curve','手臂弯过头顶','手臂弯曲成头顶上方的弧形，保持自体间隙；不是举直手。','pose','playful'),
('wave','挥手一次','抬起的前臂完成一次横向往返后回到进入姿态，用于招呼或告别。','cycle','greeting'),
('beckon','招手一次','前臂完成一次朝自身方向的招回弧线并回到进入姿态，用于示意靠近。','cycle','invite'),
('shoo','向外摆手一次','前臂完成一次背离自身方向的送出弧线并回到进入姿态，用于婉拒或示意退开。','cycle','decline'),
]
for side,cn in [('left','左'),('right','右')]:
    for suffix,name,effect,kind,tag in arm_defs:
        display_name={'rest':cn+'臂自然放下','raise':'举起'+cn+'手','forward':cn+'臂向前伸','side':cn+'臂向外侧伸','chest':cn+'手收在胸前','hip':cn+'手叉腰姿态','cheek':cn+'手托腮姿态','salute':cn+'手敬礼姿态','overhead_curve':cn+'臂弯过头顶','wave':cn+'手挥手一次','beckon':cn+'手招手一次','shoo':cn+'手向外摆手一次'}[suffix]
        add('单臂',f'arm_{side}_{suffix}',display_name,cn+'侧：'+effect,kind,
            f'arm:{side}',f'arm_pose:{side}',f'arm,{side},{tag}',
            ['该侧手臂已进入本手势的合法起始包络；不自动补插其他姿态动作。'] if kind=='cycle' else [])
hand_defs=[
('open','摊开手掌','五指展开成开放手掌，不改变手臂位置。','open'),
('fist','握空拳','五指屈曲成空拳，不夹持任何物品，不挥拳。','fist'),
('index','伸食指','仅食指伸直，其余手指收拢；指向沿当前手部轴线，不自动定位目标。','point'),
('thumb_up','竖拇指','仅拇指伸出并朝上，四指收拢，呈点赞手势；手臂位置不变。','praise'),
('thumb_down','倒拇指','仅拇指伸出并朝下，四指收拢，呈不赞同手势；手臂位置不变。','disagreement'),
('v','比V手势','食指和中指伸开成V形，其余三指收拢。','celebrate'),
('ok','比OK手势','拇指与食指形成圆环，其余三指伸开；无物体位于指间。','agreement'),
('finger_heart','比指尖心','拇指与食指交叉形成小心形轮廓，其余三指收拢；不夹取物品。','affection'),
('three','比三手势','拇指、食指、中指伸出，无名指与小指收拢，明确采用这一种三指造型。','count'),
('four','比四手势','拇指收拢，其余四指伸出。','count'),
('pinky','伸小指','仅小指伸出，其余四指收拢，用于隔空拉钩示意；不与人接触。','promise'),
('horns','摇滚手势','食指和小指伸出，中指、无名指与拇指收拢；不摆动手臂。','music'),
('crossed','交叉双指','食指和中指交叉，其余手指收拢，呈祝好运手势。','wish'),
('pinch','空捏手势','拇指与食指指腹相对，其余三指收拢，形成空捏轮廓；不是比心，也不夹取物品。','small'),
]
for side,cn in [('left','左'),('right','右')]:
    for suffix,name,effect,tag in hand_defs:
        resources=f'hand:{side}'
        caps=f'finger_pose:{side}'
        if suffix in ('thumb_up','thumb_down'):
            resources+=f',wrist:{side}'
            caps+=f',wrist_orient:{side}'
        add('手指手势',f'hand_{side}_{suffix}',cn+'手'+name,cn+'手'+effect,'pose',resources,caps,
            f'hand,{side},{tag}',['手部存在已标定的对应手指自由度；不可用一块固定掌板冒充该手势。'])
for side,cn in [('left','左'),('right','右')]:
    add('手腕',f'wrist_{side}_up',cn+'手掌朝上',cn+'侧前臂和手腕调整到掌面朝上，手指形状和手臂端点位置保持不变。',
        'pose',f'wrist:{side}',f'wrist_orient:{side}',f'wrist,{side},offer',
        ['当前手臂姿态允许掌面翻转，且已标定翻转路径。'])
    add('手腕',f'wrist_{side}_down',cn+'手掌朝下',cn+'侧前臂和手腕调整到掌面朝下，手指形状和手臂端点位置保持不变。',
        'pose',f'wrist:{side}',f'wrist_orient:{side}',f'wrist,{side},settle',
        ['当前手臂姿态允许掌面翻转，且已标定翻转路径。'])
# A bilateral pose is a single synchronized target, not two independently scheduled skills.
bilateral=[
('open','双臂张开','双臂在身体两侧张开，形成欢迎轮廓；保持空中，不等待或触碰真人。','pose','welcome',False),
('hug','环抱姿态','双臂在身前弯成环形，手掌之间留出空间，表示隔空拥抱；不执行夹抱。','pose','comfort',False),
('cross','交叉抱臂','双前臂交叉停在胸前外侧，保持与身体及另一手臂的间隙。','pose','serious',False),
('heart','双手比心','双手在胸前相对，手指共同围成心形轮廓，保持手间间隙；不改变脸部表情。','pose','affection',True),
('palms_join','双掌合十姿态','双掌在胸前相向竖立并保留间隙，形成合十轮廓；不附加鞠躬。','pose','thanks',False),
('cup','双手捧起姿态','双手在身前并排掌心向上，形成捧起轮廓；手中为空，不承载物品。','pose','offer',False),
('cover_eyes','双手遮眼姿态','双手停在双眼前方并保持间隙，形成遮眼轮廓；不触碰屏幕或头部。','pose','shy',False),
('cover_mouth','双手捂嘴姿态','双手停在嘴部前方并保持间隙，形成捂嘴轮廓；不改变语音设备状态。','pose','surprise',False),
('head','双手抱头姿态','双手停在头部两侧附近并保持间隙，形成抱头轮廓；不压住头部。','pose','surprise',False),
('air_clap','隔空拍手一次','双掌在身前完成一次相向靠近再分开的周期；最小间距由标定包络限制，无接触拍响。','cycle','celebrate',False),
('up_v','双臂举成V形','双臂向上向外展开成一个V形姿态，手指造型保持原状。','pose','celebrate',False),
('stop','双掌向前挡住姿态','双掌在身前朝外竖立，形成停一下的手势；不推人或物体。','pose','stop',False),
]
for suffix,name,effect,kind,tag,fingers in bilateral:
    resources='arm:left,arm:right,wrist:left,wrist:right'
    caps='arm_pose:left,arm_pose:right,wrist_orient:left,wrist_orient:right,bilateral_pose'
    if suffix in ('palms_join','cup','cover_eyes','cover_mouth','air_clap','stop'):
        caps+=',palm_surface:left,palm_surface:right'
    if fingers:
        resources+=',hand:left,hand:right';caps+=',finger_pose:left,finger_pose:right'
    add('双臂协调',f'bilateral_{suffix}',name,effect,kind,resources,caps,f'bilateral,{tag}',
        ['使用一个已标定的同步姿态或单周期控制器；不得按两个独立手臂动作临时拼接。',
         '两手及手臂之间全程满足自碰撞间距；保持无接触。'] + (['进入时双手为固定掌板或已张开的可动手，手型必须持续保持；握拳等不合条件时拒绝本原子，不隐式补插张手动作。'] if suffix in ('palms_join','cup','cover_eyes','cover_mouth','air_clap','stop') else []))
block('陪伴图标','render','overlay','display_overlay',"""
icon_heart|显示爱心|在独立伴随图标区域显示一枚爱心，表示亲近；不改变眼睛或身体。|icon,affection
icon_music|显示音符|在独立伴随图标区域显示一枚音符，表示音乐氛围；不播放音乐。|icon,music
icon_sleep|显示睡眠符号|在独立伴随图标区域显示Zzz，表示困倦；不关闭系统、不进入电源睡眠。|icon,sleep
icon_question|显示问号|在独立伴随图标区域显示问号，表示疑惑；不主动发起对话或调用问答接口。|icon,curiosity
icon_sun|显示太阳|在独立伴随图标区域显示太阳，表示早晨或阳光心情；不读取时间和天气。|icon,morning
icon_moon|显示月亮|在独立伴随图标区域显示月亮，表示夜晚或安静氛围；不读取时间。|icon,night
icon_flower|显示小花|在独立伴随图标区域显示一朵小花，作为祝福或感谢的视觉表达。|icon,thanks,wish
icon_pause|显示暂停符号|在独立伴随图标区域显示两条竖线，表示稍等；不暂停其他模块或硬件。|icon,wait
""")

CAPS={
'head_pan':'头部左右转动', 'head_tilt':'头部俯仰', 'head_roll':'头部侧倾',
'display_eyes':'可分层绘制瞳孔与眼睑的屏幕眼睛', 'display_mouth':'独立嘴部绘制区域',
'display_brows':'独立眉毛绘制区域', 'display_cheeks':'独立脸颊绘制区域',
'display_overlay':'不遮挡脸部的独立图标区域', 'torso_pitch':'上身俯仰',
'torso_roll':'上身侧倾', 'torso_yaw':'上身旋转', 'shoulder_elevation':'肩部独立升降',
'bilateral_pose':'双臂同步姿态与自碰撞控制',
}
for side,cn in [('left','左'),('right','右')]:
    CAPS['arm_pose:'+side]=cn+'臂端点姿态控制（须逐姿态确认可达）'
    CAPS['wrist_orient:'+side]=cn+'前臂/手腕掌面朝向控制'
    CAPS['finger_pose:'+side]=cn+'手独立手指造型（须逐造型确认自由度）'
    CAPS['palm_surface:'+side]=cn+'手进入时已有掌面（固定掌板或已张开手）；不表示本原子会张手'
GROUPS=list(dict.fromkeys(r['group'] for r in ROWS))
ATOMS=[]
for i,r in enumerate(ROWS,1):
    visual=all(c.startswith('display_') for c in r['capabilities'])
    pre=['所需能力与本条具体预设均已映射并通过验证；仅声明有某类硬件还不够。']
    if not visual:
        pre+=['当前姿态至目标/轨迹存在已标定路径，满足关节限位、自碰撞距离和桌面倾覆约束。',
              '不移动底座，不抓取、触碰人或物体；动作包络内无外部障碍。']
    else: pre+=['对应绘制区域存在且布局不遮挡其他区域；图标/表情不代表功能状态。']
    pre+=r.pop('extra_preconditions')
    locks=sorted(set(r['resources']+([] if visual else ['motion_body'])))
    params={'hold_ms':'保持期由调用方指定，由已验证执行器限制；不派生新ID',
            'transition_ms':'进入/退出过渡时间由已验证执行器限制；不派生新ID'}
    if not visual: params.update(speed_scale='标定范围内速度比例；不派生新ID', amplitude_scale='标定范围内幅度比例；不改变手势拓扑')
    if r['kind']=='cycle': params.pop('hold_ms');params['repeat_count']='本原子固定一次；重复调用由外部安排，不派生新ID'
    affected=set(r['resources']) if not visual else set()
    if not visual:
        if 'torso' in affected:
            affected.update(['head','shoulders','arm:left','arm:right','wrist:left','wrist:right','hand:left','hand:right'])
        if 'shoulders' in r['resources']:
            sides=['left'] if r['code']=='shoulder_left' else ['right'] if r['code']=='shoulder_right' else ['left','right']
            for side in sides: affected.update(['arm:'+side,'wrist:'+side,'hand:'+side])
        for side in ['left','right']:
            if 'arm:'+side in affected: affected.update(['wrist:'+side,'hand:'+side])
            if 'wrist:'+side in affected: affected.update(['arm:'+side,'hand:'+side])
    a={
        'id':f'H{i:03d}','type':'atom','namespace':'home_companion_atoms128_v1',
        **r,'scope':'home_desktop_companion',
        'canonical_effect':r['code'], 'atomic_contract':{'pose':'一个姿态目标','render':'一个区域渲染状态','cycle':'一个可取消的手势周期'}[r['kind']],
        'hardware_class':'display' if visual else 'mechanical',
        'implementation_status':'unmapped','enabled_on_board':False,
        'capability_required':r['capabilities'],'preconditions':pre,
        'exclusive_resources':locks,'parameters':params,'aliases':[],
        'commanded_channels':r['resources'], 'potentially_affected_body_parts':sorted(affected),
        'coupling_status':'kinematics_unverified; affected_parts_are_conservative_not_proof_of_other_parts_stationary' if not visual else 'render_only',
        'geometry_preconditions':(['both_palms_ready_and_maintained'] if any(c.startswith('palm_surface:') for c in r['capabilities']) else []),
        'effect_lifetime':'cycle_once' if r['kind']=='cycle' else 'caller_bounded_hold',
        'terminal_state': '回到进入时合法姿态/画面；被取消时按执行器取消合同结束。' if r['kind']=='cycle' else '保持目标至保持期结束；结束和取消均交给该通道已验证的退出合同。',
        'cancel_contract':'需提供可取消实现：渲染撤回本效果；机械执行器受控停止并报告状态。当前未实现、未测响应时间。',
        'noise':'无音频输出；机械噪声未测' if not visual else '无音频输出',
        'energy':'unmeasured' if not visual else 'display_only_unmeasured',
        'compatibility_notes':'按高层指令判断可执行性；缺少实现证据单列，不等同语义冲突。TopK顺序不构成执行计划。',
        'retrieval_text':r['name']+'。'+r['description'].split('；')[0]+'。类别：'+r['group'],
    }
    ATOMS.append(a)
assert len(ATOMS)==128,len(ATOMS)
assert len({a['id'] for a in ATOMS})==128
assert len({a['name'] for a in ATOMS})==128
assert len({a['code'] for a in ATOMS})==128
assert all(c in CAPS for a in ATOMS for c in a['capability_required'])
assert not any('steps' in a for a in ATOMS)
def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
dump('atoms.json',ATOMS)
dump('capabilities.json',{'status':'hardware_inventory_unconfirmed','fallback_policy':'exclude_unmapped_or_unsupported_candidates; never silently substitute',
    'capabilities':{c:{'description':d,'board_status':'unknown','mapped_presets':[]} for c,d in CAPS.items()}})
dump('experiment-manifest.json',{
    'dataset_version':'home_companion_atoms128_v1','candidate_type':'atom','candidate_count':128,
    'skills_included':False,'run_status':'not_run','source':'assistant_authored_design_catalog',
    'semantic_scope':'household_and_desktop_companionship','board_enabled_count':0,
    'previous_results_applicable':False,
    'historical_results':'../home-companion-rag-20261022 (35 atoms + separate 20 skills; never relabel as this 128-item result)',
    'next_evaluation':'Freeze household high-level queries and compatibility judgments against this exact dictionary before ranking; report top1..5 and whole-set conflicts.',
    'candidate_sha256':hashlib.sha256((ROOT/'atoms.json').read_bytes()).hexdigest(),
})
counts=dict(Counter(a['group'] for a in ATOMS))
text='''# 家庭／桌面陪伴原子动作集合：128 项（v1）

这是动作设计词典，共 **128 个原子动作，0 个组合技能**。使用独立 H001–H128 编号；旧版 A 编号及其35项测试保持原样。本版尚未进行128候选检索测试，也没有将任何机械动作下发实物。

“原子”指一个姿态目标、一个渲染状态或一个可取消手势周期；并非一个电机的一次脉冲。双臂姿态可以包含多个关节，但只能由一个同步控制合同执行，不能夹入观察、判断、对话或多个阶段。迎接主人、安慰用户、陪读、晚安仪式属于上层场景，本词典不收录这些技能。

速度、幅度、持续时间、重复次数和同义说法不计入128。左右分支对应不同方向或执行器；静态闭眼与单次眨眼分别是持续状态和一次事件。眼睛、嘴、眉毛、脸颊与伴随图标是明确不同的绘制区域。机械手势与屏幕图案不会互相冒充。

**实现状态：全部 unmapped，板上启用数为0。** 38项是屏幕区域效果，90项是机械动作设计；当前硬件布局尚未确认。手指造型、头部侧倾、腰部、双臂等需要相应自由度，并非只有Orange Pi板卡就能完成。能力表是需求表，不是已安装硬件清单。

本表用自然语言描述输出；JSON另含能力要求、资源互斥、参数、起始姿态条件、退出/取消合同。控制通道与可能被机械带动的身体部位分开记录；例如抬肩可能带动整条手臂，转腰可能带动头和双臂，不能据控制通道推断其他部位绝对静止。双掌造型须已有固定掌板或张开的手型；不满足时拒绝本原子，不隐式加上张手动作。这里的合同是待实现规范，没有宣称实测取消延迟、限位或避碰已经完成。所有机械动作限定原地、无外部接触；隔空拍手不拍响，托腮/叉腰/抱头等保持自体间隙。

## 分类

| 分类 | 数量 |
|---|---:|
'''
for g,n in counts.items():text+=f'| {g} | {n} |\n'
text+='| 合计 | 128 |\n'
for group in GROUPS:
    text+='\n## '+group+'\n\n| ID | 原子动作 | 一次输出的定义 | 类型 | 所需能力 |\n|---|---|---|---|---|\n'
    for a in ATOMS:
        if a['group']!=group:continue
        cap='、'.join(CAPS[c] for c in a['capability_required'])
        text+=f"| {a['id']} | {a['name']} | {a['description']} | {a['atomic_contract']} | {cap} |\n"
text+='''
## Top3 / Top5 后续执行的边界

相关性按“在这一高层指令下能否做”判断；不是唯一最佳动作的候选也可能兼容。明确违反指令或冲突才是不兼容；方向、场景或先后关系缺失则单列条件项。硬件是否已实现另列字段，不用硬件缺失混淆语义评分。

多个候选不能直接同时播放。同一表情区域会相互覆盖；闭眼与睁圆眼、头左转与头右转等可以在某些场景先后发生，但也可能违背保持姿态或方向的要求。默认机械动作共用 motion_body 互斥资源，待有真实控制器再细化并发。需检查指令、起始姿态、持续时间、资源占用与执行顺序，不能把检索排名直接当成计划。

图标只改变画面。音符不播放音乐、Zzz不关机、暂停符号不停止硬件；停止指令和系统控制属于执行层，不能靠这些卡片实现。能力不足时从候选中排除，不静默用屏幕动画替换机械动作。

## 文件与复现

- `atoms.json`：完整128项结构化词典。
- `capabilities.json`：所需能力与未映射状态。
- `experiment-manifest.json`：候选规模、版本、SHA与未测试状态。
- `build_catalog.py`：Windows/WSL可重建数据与此表。
- `validate_catalog.py` / `validation.json`：结构、计数、重复与边界检查。
- `independent-audit.json` / `INDEPENDENT_AUDIT.md`：独立审计结果。

本版不包含任何组合技能。旧版35原子+20技能的延迟和TopK结果不适用于128项，不能复制过来作为新版成绩。
'''
(ROOT/'ATOMIC_ACTIONS_128.md').write_text(text)
print(json.dumps({'count':len(ATOMS),'groups':counts,'hardware_class':dict(Counter(a['hardware_class'] for a in ATOMS))},ensure_ascii=False))
