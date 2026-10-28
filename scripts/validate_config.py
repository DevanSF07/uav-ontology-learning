import argparse,json
p=argparse.ArgumentParser(); p.add_argument('config'); a=p.parse_args(); d=json.load(open(a.config))
assert isinstance(d.get('experiments'),list) and d['experiments']; ids=[x['id'] for x in d['experiments']]; assert len(ids)==len(set(ids)); print('Valid:',len(ids),'experiments')
