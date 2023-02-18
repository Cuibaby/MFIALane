import torch
source = "weights/best.pth"
target = "weights/culane756.pth"
s = torch.load(source)
t = s['net']
state = {}
for k,v in t.items():
    if 'resa' in k:
        k = k.replace('resa','mfia')
    if 'aggregator' in k:
        k = k.replace('aggregator','mfia')
    if 'heads.decoder' in k:
        k = k.replace('heads.decoder','decoder')
    if 'heads.exist' in k:
        k = k.replace('heads.exist','heads')
    if 'cbam' not in k:
        state[k] = v
    
s['net'] = state
print(state.keys())
torch.save(s,target)