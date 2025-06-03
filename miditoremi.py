import miditoolkit

# Constants
DEFAULT_RESOLUTION = 480  # Standard MIDI ticks per quarter note
POSITION_RES = 16  # Positions per bar (e.g., 16 = 16th notes)

def midi_to_remi(midi_path):
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    max_tick = max([note.end for track in midi.instruments for note in track.notes])
    
    remi_sequence = []

    # 1. Flatten all notes from all instruments (optional: pick only piano, etc.)
    all_notes = []
    for track in midi.instruments:
        for note in track.notes:
            all_notes.append(note)
    all_notes.sort(key=lambda x: x.start)

    current_bar = -1
    for note in all_notes:
        start_tick = note.start
        duration_tick = note.end - note.start

        bar = start_tick // (ticks_per_beat * 4)
        position = int(((start_tick % (ticks_per_beat * 4)) / (ticks_per_beat * 4)) * POSITION_RES)

        if bar != current_bar:
            remi_sequence.append(f'Bar')
            current_bar = bar

        remi_sequence.append(f'Position_{position}')
        remi_sequence.append(f'Note-On_{note.pitch}')
        remi_sequence.append(f'Duration_{tick_to_duration_class(duration_tick, ticks_per_beat)}')

    return remi_sequence


# Helper: Map tick duration to one of a few symbolic classes
def tick_to_duration_class(tick, ticks_per_beat):
    duration_map = {
        0.25: 0,   # 16th
        0.5: 1,    # 8th
        1.0: 2,    # quarter
        2.0: 3,    # half
        4.0: 4     # whole
    }
    beat_length = tick / ticks_per_beat
    closest = min(duration_map.keys(), key=lambda x: abs(x - beat_length))
    return duration_map[closest]

remi_sequence = midi_to_remi('CrimsonMirelands.mid')
print(remi_sequence)