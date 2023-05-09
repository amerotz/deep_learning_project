import pickle as pkl

def main():
    with open('dataset.pkl', 'rb') as f:
        data = pkl.load(f)

    tonic = {}
    mode = {}
    style = {}
    root = set()
    durations = {}
    scrap = set()
    tot = 0
    for name in data:
        tot += 1
        d = data[name]
        t = d['tonic']
        m = d['mode']
        s = d['style']
        if t in tonic:
            tonic[t] += 1
        else:
            tonic[t] = 1
        if m in mode:
            mode[m] += 1
        else:
            mode[m] = 1
        if s in style:
            style[s] += 1
        else:
            style[s] = 1

        for l in d['root']:
            root.update(l)

        for note in d['nmat']:
            dur = str(note[1]-note[0])

            if dur in ['15', '16', '13', '14', '11', '9', '10', '12', '26']:
                scrap.add(name)
            if dur in durations:
                durations[dur] += 1
            else:
                durations[dur] = 1

    print(tonic)
    print(mode)
    print(style)
    print(root)
    durations = list(durations.items())
    durations.sort(key=lambda x : x[1], reverse=True)
    print(durations)
    print(tot)
    print(len(scrap)/tot)

if __name__ == '__main__':
    main()
