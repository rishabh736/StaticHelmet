/* Decision for close-up webcam view.
   A helmet is reported when the model's Helmet score reaches the "strictness" level AND the helmet box sits around a face.
   A visible face with no such helmet around it is reported as NO HELMET.
   `info` holds the best Helmet / NoHelmet score near the face so it can be shown on screen for tuning. */
export let info={h:0,n:0};
const inside=(d,x,y)=>x>d.x&&x<d.x+d.w&&y>d.y&&y<d.y+d.h;
export function fuse(dets,faces,W,H,strict=.45){
  info={h:0,n:0};
  const helm=dets.filter(d=>d.c===0&&d.s0>=strict&&d.s0>d.s1*1.2);
  const bad=dets.filter(d=>d.c===1);
  if(!faces.length)return helm.concat(bad);              // no face visible: YOLO only, with the strictness rule
  return faces.map(f=>{
    const cx=f.x+f.w/2,cy=f.y+f.h/2;
    // box must contain the face centre, not be smaller than the face, and cover the area above the eyes
    const near=d=>inside(d,cx,cy)&&d.w*d.h>=.8*f.w*f.h&&d.y<f.y+.3*f.h;
    dets.filter(near).forEach(d=>{info.h=Math.max(info.h,d.s0);info.n=Math.max(info.n,d.s1)});
    const h=helm.filter(near).sort((a,b)=>b.s0-a.s0)[0];
    if(h)return {...h,s:h.s0};
    const b=bad.filter(d=>inside(d,cx,cy)).sort((a,b)=>b.s1-a.s1)[0];
    const w=f.w*1.35,hh=f.h*1.75,x0=Math.max(0,cx-w/2),y0=Math.max(0,f.y-.6*f.h);
    return {x:x0,y:y0,w:Math.min(W-x0,w),h:Math.min(H-y0,hh),s:b?b.s1:Math.max(f.s,.6),c:1,s0:0,s1:b?b.s1:0};
  });
}
