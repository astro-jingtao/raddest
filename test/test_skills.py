import os


from pathlib import Path
import subprocess
import sys

import raddest


class TestPackagedSkill:
    """The skill shipped as package data must be complete and discoverable."""

    SKILL_NAME = 'raddest-regression'

    def test_skill_tree_complete(self):
        skills_root = Path(raddest.__file__).parent / 'skills'
        container = skills_root / self.SKILL_NAME
        assert (container / 'skill' / 'SKILL.md').is_file()
        assert (container / 'skill' / 'CHANGELOG.md').is_file()
        assert (container / 'skill' / 'references' / 'method-guide.md').is_file()
        assert (container / 'meta.yaml').is_file()
        assert (skills_root / 'SKILL-DEPLOY.md').is_file()

    def test_skill_frontmatter_name(self):
        skill_md = (Path(raddest.__file__).parent / 'skills' / self.SKILL_NAME /
                    'skill' / 'SKILL.md').read_text(encoding='utf-8')
        assert skill_md.startswith('---\n')
        frontmatter = skill_md.split('---', 2)[1]
        assert f'name: {self.SKILL_NAME}' in frontmatter
        assert 'description:' in frontmatter

    def test_main_module_in_process(self, capsys):
        # imported inside the test: the module prints on import, which must
        # happen under capsys, not at collection time
        import raddest.skills.__main__  # noqa: PLC0415

        out = capsys.readouterr().out
        assert 'Deployment guide' in out

    def test_discovery_entry(self):
        repo_root = Path(raddest.__file__).parent.parent
        env = dict(os.environ)
        env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get('PYTHONPATH', '')
        res = subprocess.run(
            [sys.executable, '-m', 'raddest.skills'],
            capture_output=True,
            text=True,
            check=True,
            cwd='/tmp',
            env=env,
        )
        lines = res.stdout.strip().splitlines()
        assert len(lines) == 2
        assert Path(lines[0]).is_absolute()
        assert 'SKILL-DEPLOY.md' in lines[1]
