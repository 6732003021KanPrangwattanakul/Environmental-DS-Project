"""Extract monthly PM2.5 statistics from Air4Thai (PCD) station-year report PDFs.

Each report is a one-page table: 12 month rows x pollutant blocks. The PM2.5 block is
the right-most 4 columns: [max 24-h mean, min 24-h mean, days > standard / valid days, monthly mean].
Columns are identified by clustering the x-positions of value tokens, so the parser is
robust to the layout changes between report years.
"""
import re, sys, glob, os
import numpy as np, pandas as pd, pdfplumber

TOK = re.compile(r'^(N/A|#|-|\d+(\.\d+)?|\d+/\d+\**)\**$')

def _unused_lines_of(page):
    ws = [w for w in page.extract_words() if w['x0'] > 85]
    rows = {}
    for w in ws:
        key = round(w['top'] / 3)
        rows.setdefault(key, []).append(w)
    out = []
    for k in sorted(rows):
        r = sorted(rows[k], key=lambda w: w['x0'])
        toks = [w for w in r if TOK.match(w['text'])]
        if len(toks) >= 4 and len(toks) >= 0.8 * len(r):
            out.append(toks)
    return out

def cluster(xs, gap=9):
    xs = sorted(xs); groups = [[xs[0]]]
    for x in xs[1:]:
        (groups[-1].append(x) if x - groups[-1][-1] <= gap else groups.append([x]))
    return [np.mean(g) for g in groups]

def num(t):
    t = t.rstrip('*')
    try: return float(t)
    except ValueError: return np.nan

def parse(path):
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
    numx = sorted(w['x0'] for w in words if TOK.match(w['text']) and w['text'] not in ('-', '#'))
    xnum = numx[max(0, int(0.02 * len(numx)))]
    left = [w for w in words if w['x1'] < xnum + 1 and re.search('[\u0E00-\u0E7F]', w['text'])]
    std_lab = [w for w in left if 'มาตรฐาน' in w['text']]
    if not std_lab:
        raise ValueError(f'{path}: no standard row')
    std_top = std_lab[0]['top']
    # month labels: left-column words above the standard row whose line carries value tokens
    tops = []
    for w in sorted(left, key=lambda w: w['top']):
        if w['top'] >= std_top - 2: continue
        if tops and abs(w['top'] - tops[-1]) < 4: continue
        tops.append(w['top'])
    tops = tops[-12:]
    if len(tops) != 12:
        raise ValueError(f'{path}: found {len(tops)} month rows')
    rows = [[v for v in words if abs(v['top'] - t) < 4 and v['x0'] >= xnum - 3 and TOK.match(v['text'])] for t in tops]
    std_row = [v for v in words if abs(v['top'] - std_top) < 4 and v['x0'] >= xnum - 3]
    std_nums = [num(v['text']) for v in sorted(std_row, key=lambda v: v['x0'])]
    std_nums = [x for x in std_nums if not np.isnan(x)]
    std_val = std_nums[-1] if std_nums else None
    has_pm25 = std_val in (50.0, 37.5)
    centers = cluster([(w['x0'] + w['x1']) / 2 for r in rows for w in r])
    recs = []
    for m, r in enumerate(rows, start=1):
        vals = {}
        if has_pm25:
            for w in r:
                c = (w['x0'] + w['x1']) / 2
                j = int(np.argmin([abs(c - pc) for pc in centers]))
                if j >= len(centers) - 4:
                    vals[j - (len(centers) - 4)] = w['text']
        raw_days = vals.get(2, '')
        flag = '**' if raw_days.endswith('**') else ('*' if raw_days.endswith('*') else '')
        d = re.match(r'(\d+)/(\d+)', raw_days)
        recs.append(dict(month=m,
                         pm25_max24h=num(vals.get(0, 'nan')),
                         pm25_min24h=num(vals.get(1, 'nan')),
                         days_exceed=float(d.group(1)) if d else np.nan,
                         days_valid=float(d.group(2)) if d else np.nan,
                         completeness_flag=flag,
                         pm25_mean=num(vals.get(3, 'nan')),
                         raw=' | '.join(vals.get(i, '') for i in range(4))))
    return recs, std_val

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    rows, diag = [], []
    for f in sorted(glob.glob(os.path.join(src, '*.pdf'))):
        code, year = os.path.basename(f)[:-4].split('_')
        recs, std_val = parse(f)
        for r in recs:
            r.update(station_code=code, year=int(year), std_in_report=std_val, source_file=os.path.basename(f))
            rows.append(r)
        diag.append((code, year, std_val, sum(not np.isnan(r['pm25_mean']) for r in recs)))
    df = pd.DataFrame(rows)
    cols = ['station_code','year','month','pm25_mean','pm25_max24h','pm25_min24h','days_exceed','days_valid','completeness_flag','std_in_report','source_file','raw']
    df[cols].to_csv(out, index=False)
    for d in diag: print(*d)
