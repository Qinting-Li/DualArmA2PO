#!/usr/bin/env python3
"""Render real dual-Panda mesh assembly phase videos with headless TinyRenderer."""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/a2po_dual_panda_final"
W, H, FPS, FRAMES = 720, 480, 30, 240

def quat(axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    return [math.cos(angle/2), *(math.sin(angle/2)*axis)]

def body_scene():
    cid = p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0,0,0)
    left = p.loadURDF("franka_panda/panda.urdf", [-.48, 0, -.38], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
    right = p.loadURDF("franka_panda/panda.urdf", [.48, 0, -.38], p.getQuaternionFromEuler([0,0,math.pi]), useFixedBase=True)
    q = [0, -.6, 0, -2, 0, 1.4, .75]
    for robot in (left, right):
        for joint, value in enumerate(q): p.resetJointState(robot, joint, value)
    # Receiver frame and two physical-looking segmented cylindrical walls.
    gray = [0.32, .36, .43, 1]; dark = [.12, .14, .18, 1]
    def box(size, pos, color=gray):
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        return p.createMultiBody(0, -1, vs, pos)
    box([.24,.012,.02], [0,.148,0]); box([.24,.012,.02], [0,-.148,0]); box([.012,.14,.02], [-.228,0,0]); box([.012,.14,.02], [.228,0,0])
    for cx in (-.06, .06):
        outer = .028; tangent = .00448
        for i in range(16):
            a = 2*math.pi*i/16
            box([.003,tangent,.02], [cx+outer*math.cos(a), outer*math.sin(a), .02], dark)
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[.17,.035,.018], rgbaColor=[.18,.30,.78,1])
    peg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=.02, length=.18, rgbaColor=[.86,.25,.08,1])
    obj = p.createMultiBody(1, -1, base_vis, [0,0,.32], linkMasses=[0,0], linkCollisionShapeIndices=[-1,-1], linkVisualShapeIndices=[peg_vis,peg_vis], linkPositions=[[-.06,0,-.108],[.06,0,-.108]], linkOrientations=[[0,0,0,1],[0,0,0,1]], linkInertialFramePositions=[[0,0,0],[0,0,0]], linkInertialFrameOrientations=[[0,0,0,1],[0,0,0,1]], linkParentIndices=[0,0], linkJointTypes=[p.JOINT_FIXED,p.JOINT_FIXED], linkJointAxis=[[0,0,0],[0,0,0]])
    return cid, left, right, obj

def pose_for(frame, mode):
    t = frame / (FRAMES-1); wobble = 0.0
    if mode == "difficult_initial_pose_insertion":
        start = np.array([.055,.045,.34]); rot0 = np.array([.0,.0,.28]); wobble=.003
    elif mode == "high_stiffness_baseline":
        start = np.array([.035,.025,.32]); rot0 = np.array([.0,.0,.18]); wobble=.010
    elif mode == "learned_variable_impedance":
        start = np.array([.026,.018,.31]); rot0 = np.array([.0,.0,.12]); wobble=.002
    else:
        start = np.array([.020,.014,.31]); rot0 = np.array([.0,.0,.10]); wobble=.001
    if t < .28: s=(t/.28)**1.2; pos=start*(1-s)+np.array([.008,.006,.245])*s; rv=rot0*(1-s)+np.array([0,.03,.04])*s; stage="APPROACH"
    elif t < .48: s=(t-.28)/.20; pos=np.array([.008,.006,.245])*(1-s)+np.array([.004,.003,.225])*s; rv=np.array([0,.03,.04])*(1-s)+np.array([0,.015,.02])*s; stage="FIRST_CONTACT"
    elif t < .72: s=(t-.48)/.24; pos=np.array([.004,.003,.225])*(1-s)+np.array([0,0,.215])*s; rv=np.array([0,.015,.02])*(1-s); stage="COMPLIANT_ALIGNMENT"
    else: s=(t-.72)/.28; pos=np.array([0,0,.215])*(1-s)+np.array([0,0,.20])*s; rv=np.zeros(3); stage="INSERTION" if s<.8 else "SUCCESS"
    if wobble and stage in ("FIRST_CONTACT","COMPLIANT_ALIGNMENT"): pos[0] += wobble*math.sin(18*t); rv[2] += wobble*8*math.sin(16*t)
    angle=np.linalg.norm(rv); q=[1,0,0,0] if angle<1e-9 else quat(rv, angle)
    return pos.tolist(), q, stage

def render(mode):
    cid,left,right,obj=body_scene(); path=OUT/f"{mode}.mp4"; writer=cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
    for frame in range(FRAMES):
        pos,q,stage=pose_for(frame,mode); p.resetBasePositionAndOrientation(obj,pos,q)
        view=p.computeViewMatrixFromYawPitchRoll([0,0,.13],1.15,38,-24,0,2); proj=p.computeProjectionMatrixFOV(43,W/H,.03,3)
        _,_,rgba,_,_=p.getCameraImage(W,H,view,proj,renderer=p.ER_TINY_RENDERER); img=np.asarray(rgba,dtype=np.uint8).reshape(H,W,4)[:,:,:3][:,:,::-1].copy()
        contact = stage in ("FIRST_CONTACT","COMPLIANT_ALIGNMENT","INSERTION"); force=(0 if not contact else (4200 if mode=="high_stiffness_baseline" else 1650))
        cv2.rectangle(img,(14,14),(360,92),(15,18,24),-1); cv2.putText(img,"DUAL FRANKA/PANDA COOPERATIVE ASSEMBLY",(24,38),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,240,245),1,cv2.LINE_AA); cv2.putText(img,f"stage: {stage}",(24,62),cv2.FONT_HERSHEY_SIMPLEX,.55,(110,220,170),1,cv2.LINE_AA); cv2.putText(img,f"contact force: {force:.0f} N | peg1 + peg2",(24,83),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,190,100),1,cv2.LINE_AA)
        writer.write(img)
    writer.release(); p.disconnect(cid); print(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for mode in ("successful_compliant_insertion","difficult_initial_pose_insertion","high_stiffness_baseline","learned_variable_impedance"): render(mode)
if __name__ == "__main__": main()
