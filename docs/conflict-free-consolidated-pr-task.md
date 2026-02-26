# Task: Create a conflict-free consolidated PR for provider runtime config + contract-validation tests

## Goal
Produce one PR that includes all required updates from:
1. "Add runtime configuration for AI providers"
2. "Update backend tests for contract validation"

without merge conflicts.

## Recommended approach
1. Create a fresh branch from the latest base branch (typically `main`).
2. Cherry-pick the **functional commits** (not the old merge commits) from both original branches in a deterministic order:
   - provider/runtime configuration changes first
   - contract-validation test updates second
3. Resolve any overlap by keeping:
   - env-driven provider config initialization in backend services/app startup
   - stricter API/narration contract assertions in backend tests
4. Run backend tests and fix any semantic mismatches.
5. Open a single PR with both updates.

## Concrete command sequence
```bash
# from up-to-date main
git checkout main
git pull
git checkout -b chore/consolidate-provider-config-and-contract-tests

# example: use commit SHAs from your repo history
# provider/runtime config commits
git cherry-pick <provider_commit_sha_1> [<provider_commit_sha_2> ...]

# contract validation test commits
git cherry-pick <contract_test_commit_sha_1> [<contract_test_commit_sha_2> ...]

# if conflicts occur, edit files, then:
git add <resolved_files>
git cherry-pick --continue

# verify
git status

# run tests (example)
cd backend
pytest
```

## Conflict policy (when same files overlap)
- `backend/app/services/llm_client.py`: keep runtime env configurability (`*_API_KEY`, model, endpoint, timeout, retries).
- `backend/app/main.py`: keep provider initialization from env-backed settings.
- `backend/tests/test_api.py`: keep response contract checks (`trace_id`, single paragraph constraints, 250–350 words).
- `backend/tests/test_narration.py`: keep regeneration/validation behavior and safety-branch expectations.

## PR acceptance checklist
- [ ] Provider runtime config is environment-driven and backward-compatible defaults are present.
- [ ] `/generate` and `/deepen` response contracts remain stable.
- [ ] Contract tests pass with updated constraints.
- [ ] No conflict markers or dropped assertions remain.
- [ ] PR description explains this is a consolidated replacement for the two conflicting PRs.
