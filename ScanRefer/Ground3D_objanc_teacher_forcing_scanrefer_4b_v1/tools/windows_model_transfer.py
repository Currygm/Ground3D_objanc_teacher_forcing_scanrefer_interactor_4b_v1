"""Official model transfer only: F-drive staging, verified ranges, SHA256, SCP.
No local model execution. After both repositories transfer, run CPU-only checks on224.
"""
import hashlib,json,re,subprocess,time,urllib.request,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path,PurePosixPath
ROOT=Path(r'F:\lianggm2\scientificResearch\research\model_transfer\ground3d_4b')
PROJECT='/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_4b_v1'
REPOS=[('amolharsh/Ground3D-LMM-4B-3D','ground3d_3d'),('Qwen/Qwen3-VL-4B-Instruct','qwen3_vl_4b')]
CHUNK=8*1024*1024
LOCK=threading.Lock()
def log(*args):
    with LOCK:print(*args,flush=True)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def valid_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_.\-/]+',name) or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('Invalid repository file name')
def migrate_old_ranges(path,chunks):
    old=path.with_name(path.name+'.chunks')
    if not old.exists():return
    for source in old.iterdir():
        if not source.name.isdigit():continue
        start=int(source.name)
        with source.open('rb') as f:
            position=start
            while data:=f.read(CHUNK):
                target=chunks/str(position)
                if not target.exists() or target.stat().st_size<len(data):target.write_bytes(data)
                position+=len(data)
                if len(data)<CHUNK:break

def download_ranges(url,path,size):
    chunks=ROOT/'ranges8m'/path.parent.name/path.name;chunks.mkdir(parents=True,exist_ok=True)
    migrate_old_ranges(path,chunks)
    recovered=0
    for temp in chunks.glob('*.download'):
        part=temp.with_suffix('');header=temp.with_suffix('.headers')
        ranges=re.findall(r'content-range: bytes (\d+)-(\d+)/(\d+)',header.read_text(errors='replace').lower()) if header.exists() else []
        if ranges and part.name.isdigit():
            start,end,total=map(int,ranges[-1]);done=part.stat().st_size if part.exists() else 0
            if start==int(part.name)+done and total==size and temp.stat().st_size<=end-start+1:
                with part.open('ab') as output,temp.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''):output.write(block)
                recovered+=1
        temp.unlink()
        if header.exists():header.unlink()
    if recovered:log('RECOVERED_INTERRUPTED_RANGES',path.name,recovered)
    def worker(start):
        end=min(size,start+CHUNK)-1;part=chunks/str(start)
        for attempt in range(200):
            done=part.stat().st_size if part.exists() else 0
            if done==end-start+1:return part
            if done>end-start+1:raise ValueError('Oversized staged range')
            temp=chunks/(str(start)+'.download');header=chunks/(str(start)+'.headers')
            try:
                result=subprocess.run(['curl.exe','--location','--range',f'{start+done}-{end}','--max-time','90','--silent','--show-error','--output',str(temp),'--dump-header',str(header),'--write-out','%{http_code}',url+f'?download=true&range={start}&offset={done}'],capture_output=True,text=True,timeout=110)
                ranges=re.findall(r'content-range: bytes (\d+)-(\d+)/(\d+)',header.read_text(errors='replace').lower()) if header.exists() else []
                if result.stdout.strip()!='206' or not ranges or tuple(map(int,ranges[-1]))!=(start+done,end,size):raise ValueError('Invalid byte-range response')
                if temp.exists():
                    if temp.stat().st_size>end-start-done+1:raise ValueError('Range response oversized')
                    with part.open('ab') as output,temp.open('rb') as source:
                        for block in iter(lambda:source.read(1024*1024),b''):output.write(block)
                if attempt==0 or attempt%10==0:log('RANGE_PROGRESS',path.name,start,part.stat().st_size,'/',end-start+1)
            except Exception as exc:log('RETRY',path.name,start,type(exc).__name__);time.sleep(2)
            finally:
                if temp.exists():temp.unlink()
                if header.exists():header.unlink()
        raise RuntimeError('Range retries exhausted '+path.name+' '+str(start))
    starts=list(range(0,size,CHUNK))
    with ThreadPoolExecutor(max_workers=64) as executor:list(executor.map(worker,starts))
    partial=path.with_name(path.name+'.verified-download')
    with partial.open('wb') as output:
        for start in starts:
            with (chunks/str(start)).open('rb') as source:
                for b in iter(lambda:source.read(8*1024*1024),b''):output.write(b)
    return partial

def download_small(url,path,size):
    part=path.with_name(path.name+'.download')
    for attempt in range(30):
        result=subprocess.run(['curl.exe','--location','--continue-at','-','--max-time','120','--silent','--show-error','--output',str(part),url],capture_output=True,text=True,timeout=140)
        if result.returncode==0 and part.stat().st_size==size:return part
        log('SMALL_RETRY',path.name,attempt);time.sleep(2)
    raise RuntimeError('Small-file transfer incomplete '+path.name)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for repo,folder in REPOS:
        dest=ROOT/folder;dest.mkdir(exist_ok=True);metadata_path=dest/'source_metadata.json'
        if metadata_path.exists():meta=json.loads(metadata_path.read_text())
        else:
            with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=60) as response:meta=json.load(response)
            metadata_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        revision=meta['sha'];manifest={'repo_id':repo,'revision':revision,'files':{}}
        for info in meta['siblings']:
            name=info['rfilename'];valid_name(name)
            if name=='.gitattributes' or name.startswith(('images/','assets/')):continue
            size=info.get('size',info.get('lfs',{}).get('size'));expected=info.get('lfs',{}).get('sha256')
            path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists() or path.stat().st_size!=size or (expected and digest(path)!=expected):
                log('DOWNLOAD',repo,name,size)
                url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name
                part=download_ranges(url,path,size) if size>256*1024*1024 else download_small(url,path,size)
                if part.stat().st_size!=size or (expected and digest(part)!=expected):raise RuntimeError('Official size/SHA256 mismatch '+name)
                part.replace(path)
            record={'size':size,'sha256':digest(path),'official_lfs_sha256':expected}
            # Non-LFS files also match the official Git blob, not only our transfer hash.
            if not expected and info.get('blobId'):
                with path.open('rb') as file:data=file.read()
                git_hash=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                if git_hash!=info['blobId']:raise RuntimeError('Official Git blob mismatch '+name)
                record['official_git_blob_sha1']=git_hash
            manifest['files'][name]=record;log('VERIFIED',repo,name)
        manifest_path=dest/'download_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        remote=PROJECT+'/models/'+folder
        subprocess.run(['ssh','224','mkdir -p '+remote],check=True)
        for name in manifest['files']:
            parent=str(PurePosixPath(name).parent)
            if parent!='.':subprocess.run(['ssh','224','mkdir -p '+remote+'/'+parent],check=True)
            existing=subprocess.run(['ssh','224','sha256sum '+remote+'/'+name],capture_output=True,text=True,timeout=600)
            if existing.returncode==0 and existing.stdout.split()[0]==manifest['files'][name]['sha256']:
                log('REMOTE_VERIFIED_SKIP_TRANSFER',folder,name)
            else:
                log('SCP_BEGIN',folder,name)
                subprocess.run(['scp',str(dest/name),'224:'+remote+'/'+name],check=True)
        for path in [metadata_path,manifest_path]:subprocess.run(['scp',str(path),'224:'+remote+'/'+path.name],check=True)
        log('TRANSFERRED',folder)
    log('CPU_FULL_INITIALIZATION_BEGIN')
    subprocess.run(['ssh','224','bash '+PROJECT+'/tools/initialize_after_transfer.sh'],check=True)
    (ROOT/'transfer_complete.json').write_text(json.dumps({'status':'verified_transferred_cpu_initialized','repositories':REPOS,'project':PROJECT},indent=2))
    log('COMPLETE verified transfer and CPU initialization; no GPU training')
if __name__=='__main__':main()
