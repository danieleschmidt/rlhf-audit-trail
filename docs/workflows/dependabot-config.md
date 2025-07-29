# Dependabot Configuration

Create `.github/dependabot.yml` to automate dependency updates:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    reviewers:
      - "team-leads"
    assignees:
      - "security-team"
    commit-message:
      prefix: "deps"
      include: "scope"
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

## Auto-Approval Workflow

Create `.github/workflows/dependabot-auto-approve.yml`:

```yaml
name: Dependabot Auto-Approve

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-approve:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    
    steps:
    - name: Dependabot metadata
      id: metadata
      uses: dependabot/fetch-metadata@v1
      with:
        github-token: "${{ secrets.GITHUB_TOKEN }}"
    
    - name: Auto-approve minor updates
      if: steps.metadata.outputs.update-type == 'version-update:semver-minor' || steps.metadata.outputs.update-type == 'version-update:semver-patch'
      run: gh pr review --approve "$PR_URL"
      env:
        PR_URL: ${{ github.event.pull_request.html_url }}
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Enable auto-merge
      if: steps.metadata.outputs.update-type == 'version-update:semver-patch'
      run: gh pr merge --auto --merge "$PR_URL"
      env:
        PR_URL: ${{ github.event.pull_request.html_url }}
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Security Considerations

1. **Review Major Updates**: Always manually review major version updates
2. **Test Before Merge**: Ensure CI passes before auto-approval
3. **Monitor Dependencies**: Regular security audits of dependency changes
4. **Rollback Plan**: Maintain ability to quickly revert problematic updates