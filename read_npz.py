import numpy as np
ac = np.load('/home/lijianfeng/github/DeepMineLys/Example/input_seq.npz',allow_pickle=True)
for item in ac:
    name=np.array(item)
    Fea=ac[item].item()
    arr=np.array(Fea['avg'])
    arr=np.reshape(arr,(1,-1))
    arr-=np.min(arr)
    arr/=(np.max(arr)-np.min(arr))
    arr=np.around(arr,4)
    Arr=np.c_[name,arr]
    with open("/home/lijianfeng/github/DeepMineLys/Example/input_seq_tape.csv","a+") as of:
        np.savetxt(of,Arr, delimiter=",",fmt='%s')
        