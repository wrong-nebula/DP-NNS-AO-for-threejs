"""Trace primary camera rays and cosine-weighted ambient visibility rays."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
RADIUS=1.12
WIDTH,HEIGHT=1280,720
STRIDE=4
EYE=np.array([18.,-4,7.3])
TARGET=np.array([10.8,8.8,1.4])

def unit(v):
    return v/np.linalg.norm(v,axis=-1,keepdims=True)

def intersection(origin,direction,box):
    c,h=box[:3],box[3:]
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(c-h-origin)/direction
        b=(c+h-origin)/direction
    lower=np.minimum(a,b)
    return lower.max(axis=-1),np.maximum(a,b).min(axis=-1),lower.argmax(axis=-1)

def trace(samples):
    boxes=np.array(json.loads((ROOT/'archive.json').read_text()))
    x,y=np.meshgrid(np.arange(2,WIDTH,STRIDE),np.arange(2,HEIGHT,STRIDE))
    forward=unit(TARGET-EYE)
    right=unit(np.cross(forward,[0,0,1]))
    up=np.cross(right,forward)
    scale=np.tan(np.deg2rad(37/2))
    ray=unit(forward+(x.ravel()[:,None]+.5-WIDTH/2)*2/HEIGHT*scale*right+
             (HEIGHT/2-y.ravel()[:,None]-.5)*2/HEIGHT*scale*up)
    t=np.full(len(ray),np.inf)
    axis=np.zeros(len(ray),int)
    surface=np.full(len(ray),-1)
    for i,box in enumerate(boxes):
        near,far,a=intersection(EYE,ray,box)
        hit=(near>=.1)&(far>=near)&(near<t)
        t[hit],axis[hit],surface[hit]=near[hit],a[hit],i
    visible=surface>=0
    p=EYE+ray[visible]*t[visible,None]
    a=axis[visible]
    n=np.eye(3)[a]*(-np.sign(ray[visible][np.arange(len(a)),a]))[:,None]
    tangent=np.eye(3)[(a+1)%3]
    bitangent=np.cross(n,tangent)
    origin=p+n*.002
    u=(np.arange(samples)+.5)/samples
    phi=np.arange(samples)*np.pi*(3-np.sqrt(5))+.013
    directions=(tangent[:,None,:]*(np.sqrt(u)*np.cos(phi))[None,:,None]+
                bitangent[:,None,:]*(np.sqrt(u)*np.sin(phi))[None,:,None]+
                n[:,None,:]*np.sqrt(1-u)[None,:,None])
    hits=np.zeros((len(p),samples),bool)
    for box in boxes:
        candidates=np.flatnonzero(np.all(np.abs(origin-box[:3])<=box[3:]+RADIUS,axis=1))
        for ids in np.array_split(candidates,max(1,(len(candidates)+127)//128)):
            if not len(ids):continue
            near,far,_=intersection(origin[ids,None,:],directions[ids],box)
            hits[ids]|=np.maximum(near,0)<=np.minimum(far,RADIUS)
    ao=np.full(len(ray),np.nan)
    ao[visible]=1-hits.mean(axis=1)
    shape=x.shape
    return ao.reshape(shape),surface.reshape(shape),axis.reshape(shape)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--rays',type=int,default=512)
    args=parser.parse_args()
    ao,surface,axis=trace(args.rays)
    np.savez_compressed(args.output,ao=ao,surface=surface,axis=axis,rays=args.rays)
    print(f'{np.isfinite(ao).sum():,} visible pixels × {args.rays} hemisphere rays')
