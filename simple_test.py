import pretty_midi

midi = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)
notes = [60, 62, 64, 65, 67, 69, 71, 72]
start = 0
duration = 0.5

for note in notes:
    n = pretty_midi.Note(velocity=100, pitch=note, start=start, end=start + duration)
    piano.notes.append(n)
    start += duration

midi.instruments.append(piano)
midi.write("simple.mid")
print("Saved simple.mid")
