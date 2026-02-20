End the current coding session by committing all work and pushing to GitHub.

Steps:
1. EMAIL CHECK — run `git config --global user.email` and verify it equals `30461605+Ubajaj1@users.noreply.github.com`.
   - If it doesn't match, run: `git config --global user.email "30461605+Ubajaj1@users.noreply.github.com"`
   - Then rewrite all unpushed commits with the correct author: `git rebase HEAD~$(git rev-list --count origin/main..HEAD) --exec 'git commit --amend --reset-author --no-edit'`
   - Inform the user that the email was corrected so contributions will appear on GitHub.
2. Run `git status` to see what's changed
3. Show the user a summary of what will be committed
4. SAFETY CHECK — verify none of these are staged or about to be committed:
   - `.env` or any file containing API keys
   - `docs/plans/` directory
   - `greenpes_implementation_plan.md`
   - `.claude/settings.local.json`
   - Any file matching `*api_key*` or `*API_KEY*`
   If any of these appear, STOP and warn the user before proceeding.
5. First run `/update-progress` to ensure PROGRESS.md is current
6. Stage only safe files: source code, tests, PROGRESS.md, .gitignore, .env.example, CLAUDE.md, README.md, .claude/settings.json, .claude/commands/
7. Create a commit with a descriptive message summarizing what was done this session
8. Ask the user to confirm before pushing: "Ready to push to origin/main. Confirm? (yes/no)"
9. If confirmed, run `git push origin main`
10. Show the user the GitHub URL for the push

If there's nothing to commit, say so and skip to checking if a push is needed.
