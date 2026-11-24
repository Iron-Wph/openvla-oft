"""In-place JPEG publication with a per-episode write-ahead receipt."""
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from reencode_rbench_jpeg import transcode


def syncdir(d):
    fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.writing')
    with t.open('w') as f:
        json.dump(x,f);f.flush();os.fsync(f.fileno())
    t.replace(p);syncdir(p.parent)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_bytes())
def rel(r):
    n=r['dataset']
    return Path(n)/r['rlds_version']/f"{n}-{r['split']}.tfrecord-{r['shard']:05d}-of-{r['shard_count']:05d}"
def stamp(p):
    s=p.stat();return [s.st_size,s.st_mtime_ns]
def source_rows():return load(RUN/'baseline'/'manifest.json')
def journal(r):return RUN/'receipts'/rel(r).with_suffix(rel(r).suffix+'.json')
def template(r):return RUN/'baseline'/rel(r).parent
def expected_parts(r):
    info=load(template(r)/'dataset_info.json')
    return int(next(s for s in info['splits'] if s['name']==r['split'])['shardLengths'][r['shard']])
def check(r,d):
    assert d['source']==r['source'] and d['dataset']==r['dataset'] and d['version']==r['rlds_version']
    assert d['images']==2*(r['frames']+1) and d['records']==expected_parts(r)
def preflight():
    if GATE.exists():return
    assert not (RUN/'initialized.json').exists()
    p=ROOT/'manifest.json'; m=load(p); n=ROOT/'norm_stats.json'
    assert sha(n)==NORM_SHA and hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()==load(n)['manifest_sha256']
    assert m and len({str(rel(r)) for r in m})==len({r['source'] for r in m})==len(m)
    if PREV is not None:
        assert PREV.stat().st_dev==ROOT.stat().st_dev
    base=RUN/'baseline';base.mkdir(exist_ok=True)
    shutil.copyfile(p,base/p.name);shutil.copyfile(n,base/n.name)
    catalog={};reuse=[]
    checked=set()
    for r in m:
        path=ROOT/rel(r)
        assert path.is_file() and not path.is_symlink() and path.stat().st_nlink==1 and path.resolve().is_relative_to(ROOT)
        catalog[str(rel(r))]=stamp(path)
        if path.parent not in checked:
            features=load(path.parent/'features.json')
            obs=features['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']['observation']['featuresDict']['features']
            if any(obs[cam]['image']['encodingFormat']!='png' for cam in ('image','wrist_image')):
                raise ValueError(f'Expected PNG input metadata: {path.parent}')
            checked.add(path.parent)
        sub=template(r)
        if not sub.exists():
            sub.mkdir(parents=True)
            for name in ('features.json','dataset_info.json'):shutil.copyfile(path.parent/name,sub/name)
        if PREV is None:
            continue
        old=PREV/rel(r);receipt=old.with_name(old.name+'.json')
        if receipt.exists():
            d=load(receipt);check(r,d)
            assert d['source_stat']==stamp(path) and old.stat().st_size==d['out']
            reuse.append(r)
    with cf.ThreadPoolExecutor(4) as pool:
        digests=pool.map(sha,[PREV/rel(r) for r in reuse])
        for r,h in zip(reuse,digests,strict=True):
            old=PREV/rel(r);assert h==load(old.with_name(old.name+'.json'))['sha256']
    dump(RUN/'inventory.json',catalog);dump(RUN/'reuse.json',reuse)
    assert p.read_bytes()==(base/p.name).read_bytes()
    p.rename(GATE);syncdir(ROOT)
    dump(RUN/'initialized.json',{'episodes':len(m),'eligible':len(reuse),'norm_sha256':NORM_SHA})
inventory={}
def install(r,use_prior=False):
    jp=journal(r);src=ROOT/rel(r)
    stage=PREV/rel(r) if use_prior else src.with_name('.'+src.name+'.inplace-jpeg')
    receipt=stage.with_name(stage.name+'.json')
    if jp.exists():
        d=load(jp);check(r,d)
        if stamp(src)!=d['source_stat']:
            assert src.stat().st_size==d['out'] and sha(src)==d['sha256']
            return d
    assert GATE.exists() and stamp(src)==inventory[str(rel(r))]
    d=load(receipt) if use_prior else transcode((src,stage,r))
    check(r,d)
    assert d['source_stat']==stamp(src) and stage.stat().st_size==d['out'] and sha(stage)==d['sha256']
    with stage.open('rb') as f:os.fsync(f.fileno())
    d=dict(d,prior=use_prior,installed_at=time.time())
    dump(jp,d)
    assert stamp(src)==d['source_stat']
    os.replace(stage,src);syncdir(src.parent)
    if not use_prior and receipt.exists():receipt.unlink()
    return d
def task(name):
    assert GATE.exists(), 'Run prepare before starting task workers'
    (RUN/'status').mkdir(exist_ok=True)
    lock=(RUN/'status'/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sub=sorted([r for r in source_rows() if r['task_id']==name],key=lambda r:-r['frames']);assert sub
    s=dict(task=name,host=socket.gethostname(),start=time.time(),verified=0,images=0,out=0)
    path=RUN/'status'/(name+'.json');dump(path,s)
    try:
        with cf.ProcessPoolExecutor(WORKERS) as ex:
            futures=[ex.submit(install,r) for r in sub]
            for fut in cf.as_completed(futures):
                d=fut.result();s['verified']+=1;s['images']+=d['images'];s['out']+=d['out'];s['updated']=time.time();dump(path,s)
        ref=template(sub[0]);dest=(ROOT/rel(sub[0])).parent
        feats=load(ref/'features.json')
        obs=feats['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']['observation']['featuresDict']['features']
        for cam in ('image','wrist_image'):
            im=obs[cam]['image'];assert im['encodingFormat']=='png';im.update(encodingFormat='jpeg',shape={'dimensions':['256','256','3']})
        info=load(ref/'dataset_info.json')
        for split in info['splits']:
            ds=[load(journal(r)) for r in sorted(sub,key=lambda r:r['shard']) if r['split']==split['name']]
            assert [d['records'] for d in ds]==list(map(int,split['shardLengths']))
            split['numBytes']=str(sum(d['out'] for d in ds))
        dump(dest/'dataset_info.json',info);dump(dest/'features.json',feats)
        s.update(done=True,ended=time.time());dump(path,s)
    except BaseException as e:s.update(error=repr(e));dump(path,s);raise
def finalize():
    m=source_rows()
    for n in {r['task_id'] for r in m}:assert load(RUN/'status'/(n+'.json'))['done']
    def audit(r):
        d=load(journal(r));check(r,d);p=ROOT/rel(r)
        assert p.stat().st_size==d['out'] and sha(p)==d['sha256']
        return d
    with cf.ThreadPoolExecutor(6) as ex:ds=list(ex.map(audit,m))
    assert sha(ROOT/'norm_stats.json')==NORM_SHA and sha(GATE)==sha(RUN/'baseline'/'manifest.json')
    assert all(not list((ROOT/rel(r).parent).glob('.*.inplace-jpeg.partial')) for r in [next(x for x in m if x['task_id']==n) for n in {r['task_id'] for r in m}])
    GATE.replace(ROOT/'manifest.json');syncdir(ROOT)
    dump(RUN/'published.json',dict(episodes=len(ds),out=sum(d['out'] for d in ds),input=sum(d['inp'] for d in ds),images=sum(d['images'] for d in ds),norm_sha256=NORM_SHA,time=time.time()))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='RLDS dataset to replace in place')
    parser.add_argument('run', type=Path, help='separate directory for metadata and receipts')
    parser.add_argument('stage', help='prepare, finalize, or a task ID from the manifest')
    parser.add_argument('--reuse', type=Path, help='verified JPEG staging tree on the same filesystem')
    parser.add_argument('--workers', default=8, type=int)
    args = parser.parse_args()
    if not __debug__:
        parser.error('Run without Python optimization so validation assertions are enabled')
    if args.workers < 1:
        parser.error('--workers must be positive')
    ROOT, RUN = args.root.resolve(), args.run.resolve()
    PREV = args.reuse.resolve() if args.reuse else None
    if RUN == ROOT or RUN.is_relative_to(ROOT):
        parser.error('Use a separate work directory outside the dataset')
    if PREV is not None and (PREV == ROOT or PREV.is_relative_to(ROOT)):
        parser.error('Use a separate verified staging tree outside the dataset')
    GATE = ROOT / 'manifest.json.inplace-jpeg-pending'
    WORKERS = args.workers
    if args.stage == 'prepare':
        RUN.mkdir(parents=True, exist_ok=True)
        NORM_SHA = sha((RUN/'baseline'/'norm_stats.json') if GATE.exists() else ROOT/'norm_stats.json')
        preflight()
        reuse_rows = load(RUN/'reuse.json')
        if reuse_rows and PREV is None:
            parser.error('Pass the original --reuse directory when resuming preparation')
        inventory = load(RUN/'inventory.json')
        with cf.ThreadPoolExecutor(4) as ex:
            receipts = list(ex.map(lambda row: install(row, True), reuse_rows))
        dump(RUN/'reused.json', dict(count=len(receipts),
                                   freed_bytes=sum(d['inp'] for d in receipts),
                                   current_bytes=sum(d['out'] for d in receipts)))
    elif args.stage == 'finalize':
        NORM_SHA = sha(RUN/'baseline'/'norm_stats.json')
        finalize()
    else:
        if args.stage not in {row['task_id'] for row in source_rows()}:
            parser.error(f'Unknown task ID: {args.stage}')
        inventory = load(RUN/'inventory.json')
        task(args.stage)
