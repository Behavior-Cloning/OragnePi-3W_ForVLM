"""128 no-screen/no-finger companion motion designs. Generates data only."""
from pathlib import Path
from collections import Counter
import json, hashlib
P=Path(__file__).resolve().parent
ROWS=[]
def add(group,code,name,desc,kind,resources,caps,tag):
 ROWS.append(dict(group=group,code=code,name=name,description=desc,kind=kind,commanded_channels=resources.split(','),capability_required=caps.split(','),tags=tag.split(',')))
def block(group,kind,resources,caps,lines):
 for row in lines.strip().splitlines():
  code,name,desc,tag=row.split('|');add(group,code,name,desc,kind,resources,caps,tag)
block('头部','pose','head','head_pan',"""
head_left|向左转头|头部朝自身左侧保持，其他控制通道不主动改变。|head,left,attention
head_right|向右转头|头部朝自身右侧保持，其他控制通道不主动改变。|head,right,attention
""")
block('头部','pose','head','head_tilt',"""
head_up|抬头|头部向上仰，停在上仰姿态，不同时弯腰。|head,up
head_down|低头|头部向下俯，停在下俯姿态，不同时弯腰。|head,down
""")
block('头部','pose','head','head_roll',"""
head_roll_left|向左歪头|头部向自身左侧侧倾；不是向左转头。|head,left,curiosity
head_roll_right|向右歪头|头部向自身右侧侧倾；不是向右转头。|head,right,curiosity
""")
add('头部','head_center','头部回正','头部回到已标定正向中立姿态，只改变头部。','pose','head','head_neutral','head,neutral')
add('头部','head_nod','点头一次','头部沿俯仰轴完成一次往返并返回进入姿态，表示肯定回应。','cycle','head','head_tilt','head,agreement')
add('头部','head_shake','摇头一次','头部沿左右转动轴完成一次往返并返回进入姿态，表示否定回应。','cycle','head','head_pan','head,disagreement')
add('头部','head_roll_sway','歪头轻晃一次','头部沿侧倾轴完成一次左右往返，头的正前方向不作主动左右转动。','cycle','head','head_roll','head,playful')
for direction,cn in [('cw','顺时针'),('ccw','逆时针')]:
 add('头部','head_circle_'+direction,'头部'+cn+'绕动一圈','头部俯仰与左右转动同步形成一条闭合椭圆姿态轨迹；从机器人正面观察为'+cn+'，幅度受限，不是分段点头再摇头。','cycle','head','head_pan,head_tilt,head_path_sync','head,stretch')
block('腰部','pose','waist','waist_pitch',"""
waist_forward|上身前倾|腰部使上身原地前倾，底盘位置保持。|waist,forward,listening
waist_back|上身后仰|腰部使上身原地后倾，底盘位置保持。|waist,back
""")
block('腰部','pose','waist','waist_roll',"""
waist_left|上身左侧倾|腰部使上身向自身左侧倾斜，底盘位置保持。|waist,left
waist_right|上身右侧倾|腰部使上身向自身右侧倾斜，底盘位置保持。|waist,right
""")
block('腰部','pose','waist','waist_yaw',"""
waist_turn_left|上身向左转|腰部使上身向左旋转，底盘朝向保持。|waist,left,attention
waist_turn_right|上身向右转|腰部使上身向右旋转，底盘朝向保持。|waist,right,attention
""")
add('腰部','waist_center','上身回正','腰部回到已标定直立中立姿态；不同时主动回正头部或收臂。','pose','waist','waist_neutral','waist,neutral')
for axis,cn,desc in [('roll','左右摇摆','侧倾'),('pitch','前后轻摆','俯仰'),('yaw','左右扭腰','旋转')]:
 add('腰部','waist_sway_'+axis,cn+'一次','腰部沿'+desc+'轴完成一次往返并回到进入姿态；底盘保持。','cycle','waist','waist_'+axis,'waist,rhythm')
for direction,cn in [('cw','顺时针'),('ccw','逆时针')]:
 add('腰部','waist_circle_'+direction,'腰部'+cn+'环绕一次','腰部俯仰与侧倾同步形成一个闭合小幅环绕；从上方看上身参考点为'+cn+'轨迹，不旋转底盘。','cycle','waist','waist_pitch,waist_roll,waist_path_sync','waist,stretch')
# Ten distinct targets and ten single continuous gestures per arm. No wrist/finger command.
ARM_POSES=[
('rest','臂自然放下','手臂回到身体同侧的自然下垂位置。','rest'),
('raise','臂向上举起','手臂向上伸出，停在举臂姿态，不挥动。','greeting'),
('forward','臂向前伸','手臂向前伸直，停在身前空中，不接触物品。','forward'),
('side','臂向外侧伸','手臂向身体同侧外方伸直，形成侧向指示轮廓；用整条手臂指示。','direction'),
('chest','臂收在胸前','肘部弯曲，臂端停在同侧胸前附近，保持间隙。','self'),
('hip','臂叉腰姿态','肘部向外、臂端停在同侧腰部外缘附近，保持间隙，不压住身体。','confident'),
('behind','臂置于身后','手臂向后收，臂端停在背后外侧空中；不接触背部。','quiet'),
('overhead','臂弯过头顶','手臂弯曲，在头顶上方形成弧形轮廓，与头部保持间隙。','playful'),
('forearm_up','前臂竖起','上臂靠近身体侧面，肘弯曲，前臂向上竖起；与整条手臂举起不同。','response'),
('forearm_side','前臂向外展开','上臂靠近身体侧面，肘弯曲，前臂向同侧外方横向展开；与整臂侧平举不同。','uncertain'),
]
ARM_CYCLES=[
('wave','挥臂一次','在身前竖直平面内，举起的前臂完成一次横向往返并返回进入姿态。','greeting'),
('beckon','招臂一次','从身前合法起点，臂端主行程沿进入坐标系-x方向招回，回程沿另一条已给定曲线返回起点；不隐式先举臂。','invite'),
('shoo','向外送臂一次','从身前合法起点，臂端主行程沿进入坐标系+x方向送出，回程沿另一条已给定曲线返回起点；不隐式先举臂。','decline'),
('press','向下压臂一次','从身前合法起点，臂端主行程沿进入坐标系-z方向下压，沿另一条已给定回程曲线回到起点，不触物。','calm'),
('lift','向上托臂一次','从身前合法起点，臂端主行程沿进入坐标系+z方向上托，沿另一条已给定回程曲线回到起点；不承托物品、不承诺掌面朝向。','encourage'),
('circle_cw','臂顺时针画圆一次','臂端在身前竖直平面画一个闭合圆，从机器人正面观察为顺时针。','playful'),
('circle_ccw','臂逆时针画圆一次','臂端在身前竖直平面画一个闭合圆，从机器人正面观察为逆时针。','playful'),
('heart_path','臂画心形一次','臂端在身前竖直平面沿一条预先给定的闭合心形曲线运动；轨迹由外部预先给定。','affection'),
('eight_path','臂画横8字一次','臂端在身前竖直平面沿一条自交的闭合横8字曲线运动；一条轨迹，不在中途切换动作。','playful'),
('pendulum','低位前后摆臂一次','手臂在身体同侧较低位置沿前后竖直平面往返一次；与举臂横向挥手不同。','rhythm'),
]
for side,cn in [('left','左'),('right','右')]:
 for kind,defs,cap in [('pose',ARM_POSES,'arm_pose:'),('cycle',ARM_CYCLES,'arm_path:')]:
  for code,name,desc,tag in defs:
   add('单臂',f'arm_{side}_{code}',cn+name,cn+'侧：'+desc,kind,'arm:'+side,cap+side,'arm,'+side+','+tag)

BOTH_POSES=[
('rest','双臂自然下垂','双臂同步落在身体两侧的自然下垂位置。','rest'),
('open','双臂横向张开','双臂在身体两侧伸展成横向T形，不改变腰和头的控制目标。','welcome'),
('forward','双臂平行前伸','双臂平行伸向身前，彼此不交叉，不触碰物品。','offer'),
('goalpost','双臂门框姿态','双上臂向两侧伸展，双肘弯曲使前臂向上，形成门框轮廓。','cheer'),
('cross_chest','双臂胸前交叉','双肘弯曲，双前臂交叉停在胸前；交叉处保持前后间隙。','serious'),
('ring','双臂环抱姿态','双臂在身前共同弯成环形，臂端不接触，用于隔空拥抱轮廓。','comfort'),
('hip','双臂叉腰姿态','双肘向外、两个臂端停在两侧腰外，保持无接触间隙。','confident'),
('behind','双臂置于身后','双臂向后收，臂端分开停在背后两侧，不相握、不交叉接触。','quiet'),
('overhead_arch','双臂头顶拱形','双臂弯曲，在头顶上方共同形成拱形，臂端之间保留间隙。','playful'),
('up_v','双臂上举V形','双臂向上向外伸出形成V形；只改变双臂位置。','celebrate'),
('front_x','双臂前伸X形','双臂伸向身前并跨过身体中线形成X形；与屈肘贴近胸前的交叉不同。','stop'),
('forearms_level','双前臂平举','双上臂靠近身体侧面，双肘弯曲，前臂平行指向前方，呈等待接应轮廓；不承诺掌面方向。','ready'),
]
BOTH_PATHS=[
('wave_sync','双臂同步挥动一次','举起的两个前臂在身前竖直平面同相横向摆动一次；两个端点始终同向。','cycle','greeting'),
('wave_opposed','双臂反相挥动一次','举起的两个前臂在身前竖直平面反相横向摆动一次；两个端点同时向相反方向运动。','cycle','rhythm'),
('open_close','双臂开合一次','身前环形双臂同步向外展开再合回进入姿态，臂端不接触。','cycle','affection'),
('lift_sync','双臂上提一次','双臂同相沿向上的主行程运动，沿外侧回程弧线返回起点；不夹带头或腰动作。','cycle','cheer'),
('press_sync','双臂下压一次','双臂同相沿向下的主行程运动，沿外侧回程弧线返回起点；不触碰物品。','cycle','calm'),
('pendulum_opposed','双臂低位交替摆动一次','双臂在身体两侧较低位置同时作反相前后往返；底盘不移动，不生成行走步态。','cycle','rhythm'),
('circle_cw','双臂同向顺时针画圆','两个臂端各在自身身前竖直平面同步画一圈，正面观察均顺时针，不穿过另一臂。','cycle','playful'),
('circle_ccw','双臂同向逆时针画圆','两个臂端各在自身身前竖直平面同步画一圈，正面观察均逆时针，不穿过另一臂。','cycle','playful'),
('circle_out','双臂镜像向外绕圈','两个臂端沿左右镜像圆轨迹同步绕一圈；圆的最高处速度均指向身体外侧。','cycle','stretch'),
('circle_in','双臂镜像向内绕圈','两个臂端沿左右镜像圆轨迹同步绕一圈；圆的最高处速度均指向身体中线。','cycle','stretch'),
('sweep_left','双臂向左扫过','双臂同步沿一条横向连续轨迹由右侧扫向左侧，停在左侧终态，不自动返回。','stroke','direction'),
('sweep_right','双臂向右扫过','双臂同步沿一条横向连续轨迹由左侧扫向右侧，停在右侧终态，不自动返回。','stroke','direction'),
]
for code,name,desc,tag in BOTH_POSES:
 add('双臂协调','both_'+code,name,desc,'pose','arm:left,arm:right','arm_pose:left,arm_pose:right,arms_sync','bilateral,'+tag)
for code,name,desc,kind,tag in BOTH_PATHS:
 add('双臂协调','both_'+code,name,desc,kind,'arm:left,arm:right','arm_path:left,arm_path:right,arms_sync','bilateral,'+tag)
# Base paths are externally parameterized bounded trajectories, not navigation behaviors.
block('底盘','stroke','base','base_translation',"""
base_forward|向前平移|底盘沿进入时自身正前方向直线平移给定距离，朝向保持。|base,forward
base_back|向后平移|底盘沿进入时自身正后方向直线平移给定距离，朝向保持。|base,back
base_left|向左平移|底盘沿进入时自身左侧直线平移给定距离，朝向保持。|base,left
base_right|向右平移|底盘沿进入时自身右侧直线平移给定距离，朝向保持。|base,right
""")
block('底盘','stroke','base','base_yaw',"""
base_turn_left|原地向左转|底盘中心位置保持，向左转过给定角度；不按角度档位另设ID。|base,left,turn
base_turn_right|原地向右转|底盘中心位置保持，向右转过给定角度；不按角度档位另设ID。|base,right,turn
""")
block('底盘','stroke','base','base_translation,base_yaw,base_combined_path',"""
base_arc_forward_left|前进左弧转|底盘以前进速度沿瞬时自身纵轴运动，朝向随弧线向左转；整个动作是一条连续弧线。|base,forward,left
base_arc_forward_right|前进右弧转|底盘以前进速度沿瞬时自身纵轴运动，朝向随弧线向右转；整个动作是一条连续弧线。|base,forward,right
base_arc_back_left|后退左转弧|底盘以后退速度沿瞬时自身纵轴运动，同时朝向向左转；左转描述朝向变化，不含糊描述尾部方向。|base,back,left
base_arc_back_right|后退右转弧|底盘以后退速度沿瞬时自身纵轴运动，同时朝向向右转；右转描述朝向变化，不含糊描述尾部方向。|base,back,right
base_slide_left_turn_left|左移同时左转|底盘中心沿进入坐标系的左侧方向平移，同时朝向向左转；不是先平移再转身。|base,left,turn
base_slide_left_turn_right|左移同时右转|底盘中心沿进入坐标系的左侧方向平移，同时朝向向右转。|base,left,turn
base_slide_right_turn_left|右移同时左转|底盘中心沿进入坐标系的右侧方向平移，同时朝向向左转。|base,right,turn
base_slide_right_turn_right|右移同时右转|底盘中心沿进入坐标系的右侧方向平移，同时朝向向右转。|base,right,turn
""")
for direction,cn in [('cw','顺时针'),('ccw','逆时针')]:
 add('底盘','base_orbit_'+direction,'朝向固定'+cn+'圆周平移','底盘中心沿外部给定中心和半径的小圆'+cn+'平移一周，朝向始终保持进入时方向；当前位置必须已在圆上，不自动走到起点或对齐切线。','cycle','base','base_translation,base_yaw,base_planar_path','base,playful')
add('底盘','base_sway_side','原地左右轻移一次','底盘中心沿横向直线左右往返一次，朝向保持，返回进入位置。','cycle','base','base_translation','base,rhythm')
add('底盘','base_sway_fore','原地前后轻移一次','底盘中心沿纵向直线前后往返一次，朝向保持，返回进入位置。','cycle','base','base_translation','base,rhythm')
add('底盘','base_sway_yaw','原地左右转摆一次','底盘中心位置保持，朝向左右往返一次，回到进入朝向。','cycle','base','base_yaw','base,rhythm')
add('底盘','base_hold','底盘保持静止','底盘在保持期内接受零运动目标；不主动寻位、回到旧位置或调用整机紧急停止。','hold','base','base_translation,base_yaw','base,quiet')
# A whole-body target has no ordered subactions: all component targets share one pose controller.
BODY=[
('neutral','全身中立姿态','头部回正、腰部直立、双臂自然下垂，作为一个同步姿态目标。','neutral'),
('bow','鞠躬姿态','头部下俯、腰部前倾、双臂自然下垂，保持一个鞠躬轮廓，不自动附加回礼流程。','respect'),
('shrink','缩身姿态','头部下俯、腰部前倾、双臂弯曲收在胸前，形成向内收拢轮廓。','shy'),
('lean_left','向左探身姿态','腰部左侧倾、头部向左转、双臂自然下垂；不检测左侧事物。','curiosity'),
('lean_right','向右探身姿态','腰部右侧倾、头部向右转、双臂自然下垂；不检测右侧事物。','curiosity'),
('recoil_open','后仰张臂姿态','腰部后倾、头部上仰、双臂横向张开，形成向后展开的轮廓。','surprise'),
('lean_open','前倾张臂姿态','腰部前倾、头部回正、双臂横向张开，形成向前迎接的轮廓。','welcome'),
('stretch_up','向上伸展姿态','腰部后倾、头部上仰、双臂上举成V形，保持整体伸展轮廓。','stretch'),
('stretch_left','左臂侧向伸展姿态','左臂弯过头顶、腰部向右侧倾、右臂自然下垂、头部回正，形成左侧伸展轮廓。','stretch'),
('stretch_right','右臂侧向伸展姿态','右臂弯过头顶、腰部向左侧倾、左臂自然下垂、头部回正，形成右侧伸展轮廓。','stretch'),
('present_left','左向展示姿态','腰和头向左转、左臂向左侧伸出、右臂收在胸前，用整条手臂指示方向。','direction'),
('present_right','右向展示姿态','腰和头向右转、右臂向右侧伸出、左臂收在胸前，用整条手臂指示方向。','direction'),
('down_behind','低头背臂姿态','头部下俯、腰部直立、双臂分开放在身后，两个臂端保持分开。','shy'),
('tilt_ring_left','左歪头环臂姿态','头部左侧倾、腰部直立、双臂在身前形成环形，保持无接触间隙。','affection'),
('tilt_ring_right','右歪头环臂姿态','头部右侧倾、腰部直立、双臂在身前形成环形，保持无接触间隙。','affection'),
('lean_reach','前倾伸臂姿态','腰部前倾、头部回正、双臂平行前伸；不触摸或抓取物品。','offer'),
('up_open','抬头展臂姿态','头部上仰、腰部直立、双臂横向张开，形成上扬展开轮廓。','cheer'),
('counter_left','左转腰右转头姿态','腰部向左转、头部相对上身向右转、双臂自然下垂；不是先后转动的两个动作。','stretch'),
('counter_right','右转腰左转头姿态','腰部向右转、头部相对上身向左转、双臂自然下垂；不是先后转动的两个动作。','stretch'),
('recoil_tuck','后缩收臂姿态','腰部后倾、头部下俯、双臂弯曲收在胸前，形成向后收拢轮廓。','retreat'),
]
for code,name,desc,tag in BODY:
 # Axes are design prerequisites, never inferred from the user's generic part list.
 head_axis,waist_axis={
 'neutral':('neutral','neutral'),'bow':('tilt','pitch'),'shrink':('tilt','pitch'),
 'lean_left':('pan','roll'),'lean_right':('pan','roll'),'recoil_open':('tilt','pitch'),
 'lean_open':('neutral','pitch'),'stretch_up':('tilt','pitch'),'stretch_left':('neutral','roll'),'stretch_right':('neutral','roll'),
 'present_left':('pan','yaw'),'present_right':('pan','yaw'),'down_behind':('tilt','neutral'),
 'tilt_ring_left':('roll','neutral'),'tilt_ring_right':('roll','neutral'),'lean_reach':('neutral','pitch'),
 'up_open':('tilt','neutral'),'counter_left':('pan','yaw'),'counter_right':('pan','yaw'),'recoil_tuck':('tilt','pitch')}[code]
 head_axes=['head_'+head_axis];waist_axes=['waist_'+waist_axis]
 add('全身同步姿态','body_'+code,name,desc,'pose','head,waist,arm:left,arm:right',','.join(head_axes+waist_axes+['arm_pose:left','arm_pose:right','body_pose_sync']),'body,'+tag)

CAPS={
'head_pan':'头部左右转动轴','head_tilt':'头部俯仰轴','head_roll':'头部侧倾轴','head_neutral':'头部中立姿态预设',
'head_path_sync':'头部多轴连续轨迹控制','waist_pitch':'腰部俯仰轴','waist_roll':'腰部侧倾轴','waist_yaw':'腰部旋转轴','waist_neutral':'腰部中立姿态预设',
'waist_path_sync':'腰部多轴连续轨迹控制','arm_pose:left':'左臂姿态控制','arm_pose:right':'右臂姿态控制',
'arm_path:left':'左臂端连续轨迹控制','arm_path:right':'右臂端连续轨迹控制','arms_sync':'双臂同步控制',
'body_pose_sync':'头、腰、双臂的单目标同步姿态控制',
'base_planar_path':'保持朝向的平面连续平移路径控制','base_translation':'底盘前后左右平移','base_yaw':'底盘原地转向','base_combined_path':'底盘平移与转向同时执行连续轨迹',
}
HARDWARE={
 'user_confirmed_present':['head','arm:left','arm:right','waist','base'],
 'base_user_description':'低自由度双腿按底盘处理，可前后左右及转弯',
 'user_confirmed_absent':['eyes','display','finger_actuators'],
 'not_assumed':['independent_wrist','independent_shoulder_elevation','high_dof_leg_gaits'],
 'joint_axes_limits_and_concurrent_control':'unverified',
 'mapping_status':'unmapped','board_enabled_count':0,
}
BODY_PARTS=['head','waist','arm:left','arm:right','base']
ATOMS=[]
for i,r in enumerate(ROWS,1):
 caps=r['capability_required'];channels=r['commanded_channels']
 affected=set(channels)
 if 'waist' in channels:affected.update(['head','arm:left','arm:right'])
 if 'base' in channels:affected.update(BODY_PARTS)
 pre=['所需部件虽已由用户确认存在，具体关节轴、活动范围和本预设可达性仍须逐项验证。',
      '当前状态满足轨迹或姿态的合法进入条件；不能隐式插入额外姿态或技能。',
      '运动范围满足关节限位、自碰撞间距、外部障碍和重心约束；本目录未实现这些检查。']
 if 'base' in channels:
  pre+=['由外部给定已核验的运动空间与边界；桌面使用须具备防跌落约束，未满足时禁止移动。',
        '低自由度腿的移动映射待验证；没有要求抬腿、单脚支撑、跳跃或手臂以外的额外自由度。']
  if 'base_planar_path' in caps:pre+=['必须验证连续平面平移并保持朝向的控制能力；圆周初态已位于圆上，不能隐式补充定位或对齐动作。']
  if 'base_combined_path' in caps:pre+=['必须验证平移和转向能够同时执行；只会分步平移/转向不满足本动作，不得自动拆分代替。']
 else:pre+=['本动作期间底盘保持静止；腰部导致的机械联动不等于独立控制其他部件。']
 if any(x.startswith('arm:') for x in channels):pre+=['仅规定臂端位置/手臂轮廓，不要求手指造型、掌面朝向或独立手腕转动；保持无接触。']
 if r['group']=='全身同步姿态':pre+=['所有目标同时进入一个姿态控制合同；不得展开成先后调用的动作步骤。']
 params={'duration_ms':'由已验证运动配置约束，不派生新ID','amplitude_scale':'仅在标定包络内缩放，不派生新ID','speed_scale':'由标定范围约束，不派生新ID'}
 params['profile_id']='逐动作标定并审核的单姿态或单轨迹配置，不在线生成技能步骤'
 if r['kind'] in ('pose','hold'):params['hold_ms']='有界保持期，需显式退出合同'
 if 'base' in channels and r['kind']!='hold':params.update(distance_m='按轨迹类型使用，由外部边界约束',yaw_rad='按轨迹类型使用，不按角度档位新增动作')
 if r['code'].startswith('base_orbit_'):params.update(center_xy_m='外部给定圆心，初态须位于该圆的合法进入点',radius_m='外部边界内的已标定半径')
 if r['code'].startswith('base_arc_'):params.update(radius_m='已标定开弧半径',sweep_rad='开弧扫过角，由轨迹包络限制；不派生新ID')
 if r['kind']=='hold':
  params={k:v for k,v in params.items() if k in ['hold_ms','profile_id']}
  pre+=['进入时底盘已经停止；这是零运动保持，不负责紧急制动、纠偏寻位或恢复之前运动。']
 term={'hold':'保持期仅维持零运动命令，不纠偏回位；结束交还控制权，不自动恢复之前的运动命令。','pose':'到达并保持一个目标；保持期结束按已验证退出合同结束，不自动接入后续动作。',
       'cycle':'完成单个闭合轨迹，回到进入时的合法目标状态；机械误差容限待标定。',
       'stroke':'完成单条有向轨迹，停在该轨迹终点；不自动增加回程。'}[r['kind']]
 entry='本动作已标定配置的合法初态；不得隐式插入准备动作。'
 if r['code'].endswith(('_beckon','_shoo','_press','_lift')):entry='身前合法起点；不满足时拒绝，不隐式举臂或移动到起点。'
 if r['code'].endswith(('_wave','_wave_sync','_wave_opposed')):entry='前臂已经举起且处于本横向挥动轨迹起点；不隐式先举臂。'
 if 'pendulum' in r['code']:entry='手臂已在身体侧方低位轨迹起点；不隐式降臂。'
 if r['code']=='both_open_close':entry='双臂已在身前环形轨迹起点；不隐式先环抱。'
 if r['code'].startswith('base_orbit_'):entry='当前位置在给定圆上，当前朝向为保持朝向；不补充定位或切线对齐。'
 a={'id':f'R{i:03d}','namespace':'home_companion_atoms128_v2','type':'atom',**r,
    'scope':'home_desktop_companion','hardware_class':'mechanical','implementation_status':'unmapped','enabled_on_board':False,
    'atomic_contract':{'hold':'一个零运动保持状态','pose':'一个姿态目标','cycle':'一个闭合手势周期','stroke':'一条有向连续轨迹'}[r['kind']],
    'resource_locks':sorted(set(channels+['motion_body'])),
    'potentially_affected_body_parts':sorted(affected),'coupling_status':'conservative_unverified_kinematics',
    'preconditions':pre,'entry_pose_requirement':entry,'parameters':params,'terminal_state':term,
    'cancel_contract':'需要受控停止和状态回报；取消、减速、保持方式及响应时间均未实现或实测。',
    'repeat_count':1,'aliases':[], 'force_or_contact':'no_contact',
    'controller_contract':{'kind':('zero_motion_hold' if r['kind']=='hold' else 'single_synchronized_target' if r['kind']=='pose' else 'single_continuous_trajectory'),'sequential_subactions':False,'implementation':'not_implemented'},
    'coordinate_frame':'几何轨迹表达于进入时冻结坐标：前+x、左+y、上+z；前进/后退转弯弧的纵向速度沿瞬时底盘自身轴。头相对上身，腰相对底盘；图形顺逆方向按注明观察面。',
    'retrieval_text':r['name']+'。'+r['description'],
    'compatibility_policy':'在高层指令下明确不能做或冲突才判不兼容；缺条件单列。语义兼容性与硬件实现状态分开统计。',
 }
 ATOMS.append(a)
assert len(ATOMS)==128,len(ATOMS)
assert len({a['name'] for a in ATOMS})==128
assert len({a['code'] for a in ATOMS})==128
assert all(c in CAPS for a in ATOMS for c in a['capability_required'])
def dump(n,obj): (P/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
dump('atoms.json',ATOMS)
dump('hardware-profile.json',HARDWARE)
dump('capabilities.json',{'description':'需求与验证状态；部件存在不证明具备各轴/轨迹控制能力。','capabilities':{k:{'description':v,'status':'unverified','enabled':False} for k,v in CAPS.items()}})
sha=hashlib.sha256((P/'atoms.json').read_bytes()).hexdigest()
dump('experiment-manifest.json',{'dataset_version':'home_companion_atoms128_v2','candidate_type':'atom','candidate_count':128,'skills_included':False,'candidate_sha256':sha,'run_status':'not_run','previous_results_applicable':False,'board_execution_performed':False,'source':'assistant_authored_design_catalog','supersedes':'../home-companion-atoms128-20261022','new_namespace_prevents_silent_id_reuse':True})
counts=dict(Counter(a['group'] for a in ATOMS))
text='''# 无屏幕、无手指版本：128个家庭陪伴原子动作（v2）

已按用户提供的结构重做：**头部、双臂、腰部，双腿按可前后左右移动及转弯的底盘处理；没有眼睛、屏幕或手指。** 本版不再有显示表情、图标、手指造型、独立手腕或耸肩动作，也不依赖眼睛、嘴或脸颊定位。

共128项，使用新的R001–R128编号，0个组合技能。v1保留为历史，不再是当前硬件对应的词典。头/腰具体轴、手臂活动范围及底盘同时移动转向能力尚未确认，所以本表仍是**按部件约束设计、待映射的动作全集**，不能视为128项已经能在实物执行。

一个原子可以是一个姿态目标、一个闭合手势周期、一条有向连续轨迹或一个零运动保持状态。20项全身同步姿态是多个关节共同到达一个目标，不是顺序调用多个动作的技能；如果实现只能分步完成，就不满足这条原子合同。速度、幅度、时长与角度档位为参数，不另计动作。几何手势是预定义臂端轨迹，不需要手指，也不临时识图或规划。

所有机械参数、限位、起止状态、取消行为仍待实现。没有下发电机或底盘指令，没有跑本版128候选Top-K。旧版的精度、延迟不能移植为本版成绩。

## 分类

| 类别 | 数量 |
|---|---:|
'''
for g,n in counts.items():text+=f'| {g} | {n} |\n'
text+='| 合计 | 128 |\n'
for group in counts:
 text+='\n## '+group+'\n\n| ID | 动作 | 一次输出定义 | 类型 | 必须具备的能力（尚待验证） |\n|---|---|---|---|---|\n'
 for a in ATOMS:
  if a['group']==group:
   text+=f"| {a['id']} | {a['name']} | {a['description']} | {a['atomic_contract']} | {'、'.join(CAPS[c] for c in a['capability_required'])} |\n"
text+='''
## 执行与评估说明

默认机械动作互斥。多个Top-K候选要检查进入姿态、保持状态、相互覆盖和顺序；排名不是执行计划。“相关”按该高层指令下能否执行判断，不强制唯一最佳动作；明确禁止或冲突才是不兼容，条件缺失另列，硬件状态单列。

底盘轨迹只执行外部给定的有界运动，不包括寻人、自主导航、避障或在线路径规划。桌面移动需要已核验的活动边界与防跌落约束；本目录未实现这些功能。按腿部底盘处理不表示它已经可以同时侧移和转向；相关项目列明这一能力前提，不能静默拆成先后两个动作。

没有部件能保证其他部位绝对不动：腰会带动上身，底盘会带动全身。数据中的控制通道与可能联动的部位分开记录。所有项目unmapped、板上启用数0；下一步根据真实关节和控制器筛选、映射后，才适合做物理执行测试。

## 文件

- `atoms.json`：128项结构化动作。
- `hardware-profile.json`：用户确认的有/无部件与尚未确认的自由度。
- `capabilities.json`：逐能力待验证状态。
- `experiment-manifest.json`：版本、候选哈希与未实测状态。
- `validation.json`、`INDEPENDENT_AUDIT.md`：结构检查与独立语义审计。
- `CHANGELOG.md`：相较v1的变化；旧H编号不能直接用于新版R编号。

重建：`python3 build_catalog.py`；检查：`python3 validate_catalog.py`。修改后需要重新审计和冻结哈希。
'''
(P/'ATOMIC_ACTIONS_128.md').write_text(text)
print(json.dumps({'count':len(ATOMS),'groups':counts,'sha256':sha},ensure_ascii=False))
