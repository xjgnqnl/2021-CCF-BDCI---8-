import pickle
import os
import numpy as np
from tqdm import tqdm


#posec3d
with open('./score_test_b/Posec3d/score_B_keypoint.pkl','rb') as posec3d_joint:
    score_posec3d_joint = pickle.load(posec3d_joint)
#posec3d

#agcn
with open('./score_test_b/AGCN/score_agcn_joint.pkl', 'rb') as agcn_joint:
    score_joint_agcn = pickle.load(agcn_joint)
with open('./score_test_b/AGCN/score_agcn_bone.pkl', 'rb') as agcn_bone:
    score_bone_agcn = pickle.load(agcn_bone)
with open('./score_test_b/AGCN/score_agcn_concate.pkl', 'rb') as agcn_concate_j_b_mj:
    score_concate_j_b_mj_agcn = pickle.load(agcn_concate_j_b_mj)
with open('./score_test_b/AGCN/score_agcn_motion.pkl', 'rb') as agcn_motion_j:
    score_motion_j_agcn = pickle.load(agcn_motion_j)
#agcn

#stgcn
with open('./score_test_b/STGCN/score_stgcn_joint.pkl', 'rb') as stgcn_joint:
    score_joint_stgcn = pickle.load(stgcn_joint)
with open('./score_test_b/STGCN/score_stgcn_bone.pkl', 'rb') as stgcn_bone:
    score_bone_stgcn = pickle.load(stgcn_bone)
with open('./score_test_b/STGCN/score_stgcn_concate.pkl', 'rb') as stgcn_concate:
    score_concate_stgcn = pickle.load(stgcn_concate)
with open('./score_test_b/STGCN/score_stgcn_motion.pkl', 'rb') as stgcn_motion:
    score_motion_stgcn = pickle.load(stgcn_motion)
#stgcn

#11-22 added model
with open('./score_test_b/Posec3d/score_B_label_smooth.pkl','rb') as label_smooth_joint:
    score_label_smooth = pickle.load(label_smooth_joint)
'''
with open('/home/xingkai/CCF-BDCI/posec3d/wuhao/score_B_chadian.pkl','rb') as chadian_joint:
    score_chadian = pickle.load(chadian_joint)
'''
with open('./score_test_b/Posec3d/score_B_clean.pkl','rb') as clean_joint:
    score_clean = pickle.load(clean_joint)

f = open('./submit/submission.csv' , 'w')
f.write('sample_indexpose,predict_categorypose' + '\n')
for i in tqdm(range(len(score_posec3d_joint))):
    #posec3d
    joint_posec3d = score_posec3d_joint[i]
    #posec3d

    #agcn
    joint = score_joint_agcn[i]
    bone = score_bone_agcn[i]
    concate_j_b_mj = score_concate_j_b_mj_agcn[i]
    motion_j = score_motion_j_agcn[i]
    #agcn

    #stgcn
    joint_stgcn = score_joint_stgcn[i]
    bone_stgcn = score_bone_stgcn[i]
    concate_stgcn = score_concate_stgcn[i]
    motion_stgcn = score_motion_stgcn[i]
    #stgcn

    joint_label_smooth = score_label_smooth[i]
    joint_clean = score_clean[i]


    score = joint_posec3d + 0.1 * (joint + bone + concate_j_b_mj + 0.2 * motion_j + joint_stgcn + bone_stgcn + concate_stgcn + 0.2 * motion_stgcn) + 0.8 * joint_label_smooth + 0.8 * joint_clean#
    label = r = np.argmax(score)
    f.write(str(i) + "," + str(label) + "\n")
f.close()


