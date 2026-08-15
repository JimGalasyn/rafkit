# Releasing

Three steps do the release. Everything else in this file is a failure mode borrowed
from the sibling repos, recorded here *before* it can happen rather than after — the
release process in `run-farm` accumulated each of these the expensive way, and there is
no reason to rediscover them.

## Before tagging

1. **The CHANGELOG must describe every commit since the last tag.** Check it, do not
   remember it:

   ```bash
   git log --oneline $(git describe --tags --abbrev=0)..main
   ```

   In `run-farm` this once turned up four of seven commits undocumented, two of which
   had changed the public surface. Write entries from the **commit messages**, not the
   diffs — the messages record why each change was made, and that does not survive
   being re-derived from a diff.

2. **Bump `version` in three places and keep them in sync:** `pyproject.toml`,
   `src/rafkit/__init__.py` (`__version__`), and `CITATION.cff`. Unlike `run-farm`,
   this package *does* export `__version__`, so it is a third place to miss.

3. **Update the README status line** if one exists. Badges are dynamic and stay
   correct on their own; a hardcoded version does not, and in `run-farm` one drifted
   through two releases unnoticed.

## Tag and publish

4. Commit, then `git tag -a vX.Y.Z`, and push the tag.
5. Publish a GitHub Release for the tag. That triggers two things you did not run:
   `.github/workflows/publish-pypi.yml` (OIDC trusted publishing to PyPI) and, if the
   repo is connected to Zenodo, the minting of a **version DOI**.

## After publishing — the step that is missed every time

6. **Record the new version DOI in `CITATION.cff` — and add its CHANGELOG line in the
   same commit.** In `run-farm`, three consecutive releases were minted and left
   unrecorded. A `CITATION.cff` that omits the DOI for the version it names is wrong in
   the one file whose entire job is to be cited correctly.

   The CHANGELOG half is not decoration. This commit lands *after* a release and is
   therefore the first commit of the *next* one, where step 1 reliably catches it as an
   undocumented change — it did so on v0.2.0, v0.3.0 **and** v0.4.0. Writing its entry
   while you are already in the file costs nothing and stops the check from spending
   its attention on the same known gap every time.

## Verify the artifacts, not the invocations

A green workflow is not evidence that anything was published. Check the artifact:

```bash
pip index versions rafkit          # PyPI actually has it
gh release view vX.Y.Z             # the release exists and has the tag
curl -s https://zenodo.org/api/records?q=rafkit | head   # the DOI was minted
```

## First publish (one-time setup)

- **PyPI**: the project does not exist yet, so add a **pending publisher** at
  pypi.org → Account → Publishing:
  - PyPI Project Name: `rafkit`
  - Owner: `JimGalasyn`
  - Repository name: `rafkit`
  - Workflow name: `publish-pypi.yml`
  - Environment name: `pypi`
- **GitHub**: create an Environment named `pypi` (Settings → Environments).
- **Zenodo**: zenodo.org → Account → GitHub → flip the switch on `JimGalasyn/rafkit`.
  Zenodo only sees releases created *after* the switch is on.
- **Codecov**: add `CODECOV_TOKEN` to repository secrets.

None of these can be done from a shell; they are all web-console steps.
