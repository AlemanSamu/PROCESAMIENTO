#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class CollectedItem:
    label: str
    source: Path
    destination: Path


def _iter_candidate_files(root: Path, names: set[str]) -> Iterable[Path]:
    for path in root.rglob('*'):
        if path.is_file() and path.name in names:
            yield path


def _score(path: Path, project_id: str) -> tuple[int, float]:
    text = str(path).lower()
    score = 0
    if project_id.lower() in text:
        score += 100
    if 'pipeline' in text:
        score += 20
    if 'final' in text or 'result' in text:
        score += 10
    return score, path.stat().st_mtime


def _pick_best(paths: list[Path], project_id: str) -> Path | None:
    if not paths:
        return None
    ranked = sorted(paths, key=lambda p: _score(p, project_id), reverse=True)
    return ranked[0]


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description='Collect final E2E evidence artifacts.')
    parser.add_argument('--project-id', required=True, help='Project id used in mobile/backend flow.')
    args = parser.parse_args()

    project_id = args.project_id.strip()
    if not project_id:
        raise SystemExit('project_id vacio')

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / 'data'
    defense_root = repo_root / 'defense_package'
    output_dir = defense_root / f'final_e2e_{project_id}'

    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        'quality_report.json',
        'colmap_report.json',
        'fallback_report.json',
        'preprocessing_manifest.json',
        'capture_metadata.json',
        'dataset_validation_report.json',
    }

    found_by_name: dict[str, list[Path]] = {name: [] for name in wanted}
    search_roots = [data_root, repo_root / 'backup_refactor', repo_root / 'tmp_pipeline_downloads']
    for root in search_roots:
        if not root.exists():
            continue
        for file in _iter_candidate_files(root, wanted):
            found_by_name[file.name].append(file)

    collected: list[CollectedItem] = []
    missing: list[str] = []

    for name in sorted(wanted):
        best = _pick_best(found_by_name[name], project_id)
        if best is None:
            missing.append(name)
            continue
        dst = output_dir / 'reports' / name
        _safe_copy(best, dst)
        collected.append(CollectedItem(label=name, source=best, destination=dst))

    glb_candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        glb_candidates.extend(root.rglob('*.glb'))
    best_glb = _pick_best(glb_candidates, project_id)
    if best_glb is not None:
        dst = output_dir / 'model' / best_glb.name
        _safe_copy(best_glb, dst)
        collected.append(CollectedItem(label='final_glb', source=best_glb, destination=dst))
    else:
        missing.append('final_glb')

    colmap_log_candidates: list[Path] = []
    for root in [repo_root, data_root]:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            n = p.name.lower()
            if ('colmap' in n and n.endswith('.log')) or n in {'log.txt', 'colmap.log'}:
                colmap_log_candidates.append(p)
    best_log = _pick_best(colmap_log_candidates, project_id)
    if best_log is not None:
        dst = output_dir / 'logs' / best_log.name
        _safe_copy(best_log, dst)
        collected.append(CollectedItem(label='colmap_log', source=best_log, destination=dst))
    else:
        missing.append('colmap_log')

    summary = {
        'project_id': project_id,
        'generated_at': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'copied': [
            {
                'label': item.label,
                'source': str(item.source),
                'destination': str(item.destination),
            }
            for item in collected
        ],
        'missing_optional_or_not_found': missing,
    }

    readme_lines = [
        f"# Final E2E Evidence - {project_id}",
        '',
        f"Generated at: {summary['generated_at']}",
        '',
        '## Copied files',
        *[
            f"- {item.label}: `{item.source}` -> `{item.destination}`"
            for item in collected
        ],
        '',
        '## Missing (not found)',
    ]
    if missing:
        readme_lines.extend([f"- {m}" for m in missing])
    else:
        readme_lines.append('- None')

    (output_dir / 'README_SUMMARY.md').write_text(
        '\n'.join(readme_lines),
        encoding='utf-8',
    )

    (output_dir / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"Evidence package generated: {output_dir}")
    print(f"Files copied: {len(collected)}")
    if missing:
        print('Missing: ' + ', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
