"""pytest가 프로젝트 루트와 src를 import 경로에 넣도록 하는 설정."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
