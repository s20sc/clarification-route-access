"""Verify local records and optional companion archive/display movies."""
from hashlib import sha256
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--videos-dir',type=Path);p.add_argument('--additional-dir',type=Path);a=p.parse_args()
r=Path(__file__).resolve().parent;m=json.loads((r/'MANIFEST.json').read_text())
def check(base,files):
    for n,h in files.items():
        assert sha256((base/n).read_bytes()).hexdigest()==h, 'Digest mismatch: '+str(base/n)
check(r,m['files'])
v=json.loads((r/'VIDEO_INDEX.json').read_text());assert len(v['videos'])==v['count']==12
assert json.loads((r/v['zero_action_case_without_video']).read_text())['video_available'] is False
if a.additional_dir:check(a.additional_dir,m['additional_original_movies'])
if a.videos_dir:check(a.videos_dir,{x['delivery_file']:x['sha256'] for x in v['videos']})
print(json.dumps({'original_evidence_files_verified':77+(4 if a.additional_dir else 0),
 'original_movie_files_verified':8+(4 if a.additional_dir else 0),
 'display_movie_files_verified':12 if a.videos_dir else 0,
 'additional_archive_check':'pass' if a.additional_dir else 'NOT RUN: use --additional-dir',
 'display_movie_check':'pass' if a.videos_dir else 'NOT RUN: use --videos-dir',
 'no_video_case':'D94 case-b4a160e7; unknown retained'},sort_keys=True))
