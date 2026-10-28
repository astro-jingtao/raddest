# Deploying the Skills Shipped with raddest

The raddest package ships Claude Agent Skills as package data, under the
`raddest/skills` directory of the installed package. This guide is for the
agent (or person) making those skills visible to Claude Code on a machine —
everything needed is in this file.

Find the installed skills root (works in any environment where raddest is
installed):

```bash
python -m raddest.skills
```

It prints the absolute skills directory and points at this guide. Paths
below refer to that printed directory as `<skills>`; each skill occupies
two layers inside it:

```text
<skills>/<name>/            container
├── skill/                  the skill proper — SKILL.md plus supporting files
└── meta.yaml               external-dependency metadata — read, never deployed
```

Only `<skills>/<name>/skill/` is deployed. `meta.yaml` stays behind: it
declares what the skill expects from its environment, and you read it
before linking.

## Deployment is a directory link

Make a skill visible to Claude Code by linking its `skill/` directory into
a skills directory:

| target | link path | points at |
| ------ | --------- | --------- |
| user level (every project) | `~/.claude/skills/<name>` | `<skills>/<name>/skill` |
| project level (one project) | `<project>/.claude/skills/<name>` | same target |

Prefer a link over a copy — a link tracks the installed package version; a
copy drifts. When a link is impossible (privilege, filesystem), copy the
directory instead and note that reinstalling or upgrading raddest will not
update the copy.

Commands — replace `<skills>` with the path printed above:

Linux / macOS:

```bash
ln -s <skills>/<name>/skill ~/.claude/skills/<name>
```

Windows (PowerShell):

```powershell
New-Item -ItemType SymbolicLink -Path "$HOME\.claude\skills\<name>" -Target "<skills>\<name>\skill"
```

Windows requires Developer Mode or the "Create symbolic links" account
right to create a symlink; a failure mentioning privilege or WinError 1314
means neither is present — enable one, or fall back to a copy:

```powershell
Copy-Item -Recurse <skills>\<name>\skill "$HOME\.claude\skills\<name>"
```

Create a directory symlink, never a Windows junction: recursive removers
walk through a junction into its target and can delete the package's
content, while a symlink is removed as the link itself.

## Before linking: read meta.yaml

`<skills>/<name>/meta.yaml` declares the skill's expectations of its
environment:

| field | meaning | what you do before deploying |
| ----- | ------- | ---------------------------- |
| `depends-on` | other skills this one hard-requires to function | deploy them together — the same link procedure for each |
| `requires` | environment requirements | per entry: `home: default` (or no `home`) → verify on this machine now (run the entry's `check:` command when present; otherwise by its `kind`); `check: manual` → verify by hand following the entry's `note`; `home: project-env` → nothing to do here — the skill verifies it inside the target project at use time; a `platform:` naming another OS → the entry does not exist here |
| `claude-md` | content the environment's CLAUDE.md should carry | land it in the CLAUDE.md at its `scope` — global → the user's `~/.claude/CLAUDE.md`, project → the target project's `CLAUDE.md`. Check for equivalent existing content first; insert only what is missing, wrapped in the marker pair below |
| `integrates-with`, `notes` | informational linkage and operator notes | read; no action |

claude-md marker pair — wrap inserted content so future readers can
recognize machine-landed blocks:

```markdown
<!-- agent-skill:<skill>/<id> -->
…content…
<!-- /agent-skill:<skill>/<id> -->
```

An unsatisfied `depends-on` or `requires` entry is a report to the user
— the user fixes the gap or consciously continues; never a silent skip.
An `install:` command on a missing requirement runs only after the user
confirms.

## Safety rules

- Show the plan first — which skills, which links, which findings —
  before creating anything.
- A real directory or file already sitting at a link destination is a
  conflict: report it, never delete it. The most you may do, with the
  user's knowledge, is rename it aside (`<name>.bak.<timestamp>`).
- The procedure is idempotent — re-running it leaves an existing correct
  link alone.

## After linking

- Verify each deployment: `<skills-dir>/<name>/SKILL.md` is reachable
  through the link, and the frontmatter `name:` equals `<name>`.
- To remove a deployment, remove the link itself — on Windows, confirm
  first that it is a symlink rather than a junction:
  `Get-Item <path> | Select-Object LinkType` should show `SymbolicLink`.
