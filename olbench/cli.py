import argparse
from .pipeline import run,collect

def main():
    p=argparse.ArgumentParser(prog='olbench'); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run'); r.add_argument('--config',required=True); r.add_argument('--input',required=True); r.add_argument('--out',required=True); r.add_argument('--gt-dir')
    c=sub.add_parser('collect'); c.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='run': run(a.config,a.input,a.out,a.gt_dir)
    else: print(collect(a.out).to_string(index=False))
