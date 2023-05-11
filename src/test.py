import muspy as mp
import pickle as pkl

def main():
    with open('dataset.pkl', 'rb') as f:
        data = pkl.load(f)

    music = mp.Music()
    track = mp.Track()
    music.tracks.append(track)

    scale = 8

    piece = data['Niko_Kotoulas__RhythmChordProg_4_G#m-F#-E-C# (vi-V-IV-II).mid']
    for n in piece['nmat']:
        note = mp.Note(time=scale*n[0], pitch=n[2], duration=int(scale*(max(n[1]-n[0], 0.5))))
        track.notes.append(note)

    music.write_midi('test_0.5.mid')

if __name__ == '__main__':
    main()
