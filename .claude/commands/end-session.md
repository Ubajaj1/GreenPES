End the current coding session by committing all work and pushing to GitHub.

Steps:
1. Run `git status` to see what's changed
2. Show the user a summary of what will be committed
3. SAFETY CHECK — verify none of these are staged or about to be committed:
   - `.env` or any file containing API keys
   - `docs/plans/` directory
   - `greenpes_implementation_plan.md`
   - `.claude/settings.local.json`
   - Any file matching `*api_key*` or `*API_KEY*`
   If any of these appear, STOP and warn the user before proceeding.
4. First run `/update-progress` to ensure PROGRESS.md is current
5. Stage only safe files: source code, tests, PROGRESS.md, .gitignore, .env.example, CLAUDE.md, README.md, .claude/settings.json, .claude/commands/
6. Create a commit with a descriptive message summarizing what was done this session
7. Ask the user to confirm before pushing: "Ready to push to origin/main. Confirm? (yes/no)"
8. If confirmed, run `git push origin main`
9. Show the user the GitHub URL for the push

If there's nothing to commit, say so and skip to checking if a push is needed.
