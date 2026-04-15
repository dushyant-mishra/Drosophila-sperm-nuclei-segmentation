print('=' * 82)
print(f'{"Metric":<25} {"batch_3":>10} {"batch_5":>10} {"batch_6":>10}  {"3->6":>8}')
print(f'{"":25} {"(loose)":>10} {"(strict)":>10} {"(balanced)":>10}  {"delta":>8}')
print('=' * 82)
data = [
    ('Tracks total',          8189, 11094,  7513),
    ('Measurements total',   17001, 17001, 17001),
    ('Long outliers',          805,   780,   914),
    ('Tortuous outliers',      122,   224,   209),
    ('Thick outliers',         209,    96,   265),
    ('Taper outliers',         651,   162,   609),
    ('Single-slice outliers', 4250,  7637,  3680),
    ('Any flagged',           5303,  8218,  4880),
]
for name, b3, b5, b6 in data:
    delta = b6 - b3
    sign  = '+' if delta > 0 else ''
    print(f'{name:<25} {b3:>10,} {b5:>10,} {b6:>10,}  {sign}{delta:>8,}')
print('=' * 82)
print()
print('Multi-slice track health:')
for label, total, singles in [('batch_3', 8189, 4250), ('batch_5', 11094, 7637), ('batch_6', 7513, 3680)]:
    multi = total - singles
    dets  = 17001 - singles
    avg   = dets / max(multi, 1)
    print(f'  {label}: {multi:,} multi-slice tracks, {dets:,} detections linked, avg {avg:.1f} dets/track')
