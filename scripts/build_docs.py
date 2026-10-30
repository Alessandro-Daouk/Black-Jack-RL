"""Compiler deux fois les documents mathématiques ; --check conserve les PDF livrés."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(check=False):
    if shutil.which('pdflatex') is None:
        raise RuntimeError('Installer pdflatex, babel-french, lmodern et les paquets LaTeX recommandés.')
    if not (ROOT/'results/v2/analysis/kl_curve.png').exists():
        raise RuntimeError('Figure KL manquante : régénérer src.audit.analyze avant les PDF finaux.')
    env = dict(os.environ)
    env.setdefault('SOURCE_DATE_EPOCH', str(int(datetime(2026,10,30,tzinfo=timezone.utc).timestamp())))
    records=[]
    with tempfile.TemporaryDirectory(prefix='blackjack-latex-') as temporary:
        for name in ('modele1','modele2'):
            for _ in range(2):
                result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
                    '-file-line-error',f'-output-directory={temporary}',f'{name}.tex'],
                    cwd=ROOT/'docs',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
                if result.returncode:
                    raise RuntimeError(result.stdout[-6000:])
            log=(Path(temporary)/f'{name}.log').read_text(encoding='utf-8',errors='replace')
            problems=[line for line in log.splitlines() if any(x in line for x in
                ('Overfull', 'undefined references', 'undefined citations', 'multiply defined'))]
            if problems:
                raise RuntimeError(f'{name}: '+ '\n'.join(problems))
            source=Path(temporary)/f'{name}.pdf'
            if not source.exists() or source.stat().st_size==0:
                raise RuntimeError(f'PDF absent : {name}')
            if not check:
                shutil.copyfile(source,ROOT/'docs'/source.name)
            records.append({'document':name,'bytes':source.stat().st_size,'checks':'passed','written':not check})
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    build(parser.parse_args().check)
