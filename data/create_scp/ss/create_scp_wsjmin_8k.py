import os


num_spks = [2, 3, 4, 5]

def write_scp(scp_path, data_dir):
    scp_file = open(scp_path,'w')
    for root, dirs, files in os.walk(data_dir):
        files.sort()
        for file in files:
            scp_file.write(file+" "+root+'/'+file)
            scp_file.write('\n')
    scp_file.close()


for num_spk in num_spks:
    partition = ['tr', 'cv', 'tt']
    scp_root = f'scp/scp_wsj0_mix_8k/{num_spk}mix'
    os.makedirs(scp_root, exist_ok=True)
    root_dir = f'/home/DB/wsj0_kmix/{num_spk}speakers/wav8k/min/'
    mix_scp, mix_dir, src_scp, src_dir = {}, {}, {}, {}
    for p in partition:
        mix_scp[p] = os.path.join(scp_root, f'{p}_mix.scp')
        mix_dir[p] = os.path.join(root_dir, p, 'mix')
        src_scp[p] = []
        src_dir[p] = []
        for i in range(num_spk):
            src_scp[p].append(os.path.join(scp_root, f'{p}_s{i+1}.scp'))
            src_dir[p].append(os.path.join(root_dir, p, f's{i+1}'))
        
    for p in partition:
        write_scp(mix_scp[p], mix_dir[p])
        for i in range(num_spk):
            write_scp(src_scp[p][i], src_dir[p][i])

    print(f"WSJ0-{num_spk}mix scp files are created.")