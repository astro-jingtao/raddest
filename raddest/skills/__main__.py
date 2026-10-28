"""Print the installed skills root and the deployment guide location."""

from pathlib import Path

root = Path(__file__).parent.resolve()
print(root)
print(f"Deployment guide: {root / 'SKILL-DEPLOY.md'}")
