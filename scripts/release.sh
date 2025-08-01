#!/bin/bash

# RLHF Audit Trail - Release Automation Script
# Handles semantic versioning, changelog generation, and release deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
RELEASE_TYPE="patch"
DRY_RUN="false"
SKIP_TESTS="false"
SKIP_BUILD="false"
PUSH_TAGS="true"
CREATE_GITHUB_RELEASE="false"
REGISTRY="${REGISTRY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
RLHF Audit Trail Release Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE        Release type: major, minor, patch [default: patch]
    -d, --dry-run         Perform a dry run without making changes
    --skip-tests          Skip running tests before release
    --skip-build          Skip building Docker images
    --no-push-tags        Don't push tags to remote repository
    --github-release      Create GitHub release (requires gh CLI)
    -r, --registry REGISTRY  Docker registry for image push
    -h, --help            Show this help message

RELEASE TYPES:
    major     Increment major version (1.0.0 -> 2.0.0)
    minor     Increment minor version (1.0.0 -> 1.1.0)
    patch     Increment patch version (1.0.0 -> 1.0.1)

EXAMPLES:
    # Create a patch release
    $0

    # Create a minor release with dry run
    $0 -t minor -d

    # Create a major release and push to registry
    $0 -t major -r ghcr.io/terragonlabs

    # Skip tests and create GitHub release
    $0 --skip-tests --github-release

ENVIRONMENT VARIABLES:
    REGISTRY            Default Docker registry
    GITHUB_TOKEN        GitHub token for creating releases
    SKIP_TESTS          Skip tests (true/false)
    SKIP_BUILD          Skip build (true/false)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            RELEASE_TYPE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --no-push-tags)
            PUSH_TAGS="false"
            shift
            ;;
        --github-release)
            CREATE_GITHUB_RELEASE="true"
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate release type
if [[ "$RELEASE_TYPE" != "major" && "$RELEASE_TYPE" != "minor" && "$RELEASE_TYPE" != "patch" ]]; then
    log_error "Invalid release type: $RELEASE_TYPE. Must be 'major', 'minor', or 'patch'"
    exit 1
fi

log_info "Starting release process..."
log_info "Release type: $RELEASE_TYPE"
log_info "Dry run: $DRY_RUN"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in a Git repository
    if ! git rev-parse --git-dir &> /dev/null; then
        log_error "Not in a Git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "There are uncommitted changes. Please commit or stash them before release."
        exit 1
    fi
    
    # Check current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" ]]; then
        log_warning "Not on main/master branch. Current branch: $CURRENT_BRANCH"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
    
    # Check for VERSION file
    if [[ ! -f "VERSION" ]]; then
        log_error "VERSION file not found"
        exit 1
    fi
    
    # Check GitHub CLI if needed
    if [[ "$CREATE_GITHUB_RELEASE" == "true" ]]; then
        if ! command -v gh &> /dev/null; then
            log_error "GitHub CLI (gh) is required for creating GitHub releases"
            exit 1
        fi
        
        if ! gh auth status &> /dev/null; then
            log_error "Not authenticated with GitHub CLI. Run 'gh auth login' first."
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Get current version
get_current_version() {
    CURRENT_VERSION=$(cat VERSION)
    log_info "Current version: $CURRENT_VERSION"
    
    # Validate version format (semantic versioning)
    if ! [[ "$CURRENT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format in VERSION file: $CURRENT_VERSION"
        log_error "Expected format: X.Y.Z (semantic versioning)"
        exit 1
    fi
}

# Calculate new version
calculate_new_version() {
    IFS='.' read -r -a version_parts <<< "$CURRENT_VERSION"
    MAJOR=${version_parts[0]}
    MINOR=${version_parts[1]}
    PATCH=${version_parts[2]}
    
    case $RELEASE_TYPE in
        major)
            ((MAJOR++))
            MINOR=0
            PATCH=0
            ;;
        minor)
            ((MINOR++))
            PATCH=0
            ;;
        patch)
            ((PATCH++))
            ;;
    esac
    
    NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
    log_info "New version: $NEW_VERSION"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        log_warning "pytest not found. Skipping tests."
        return 0
    fi
    
    # Run fast tests only for release
    if ! pytest -m "not slow" --maxfail=5 -q; then
        log_error "Tests failed. Release aborted."
        exit 1
    fi
    
    log_success "Tests passed"
}

# Update version files
update_version() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would update VERSION file to $NEW_VERSION"
        return 0
    fi
    
    log_info "Updating version files..."
    
    # Update VERSION file
    echo "$NEW_VERSION" > VERSION
    
    # Update pyproject.toml if it exists
    if [[ -f "pyproject.toml" ]]; then
        if command -v python &> /dev/null; then
            python -c "
import re

with open('pyproject.toml', 'r') as f:
    content = f.read()

# Update version in pyproject.toml
content = re.sub(r'version = [\"\']\d+\.\d+\.\d+[\"\'']', 'version = \"$NEW_VERSION\"', content)

with open('pyproject.toml', 'w') as f:
    f.write(content)
" 2>/dev/null || log_warning "Failed to update pyproject.toml version"
        fi
    fi
    
    log_success "Version files updated"
}

# Generate changelog entry
generate_changelog() {
    log_info "Generating changelog entry..."
    
    # Get commits since last tag
    LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    
    if [[ -z "$LAST_TAG" ]]; then
        log_info "No previous tags found. Generating initial changelog."
        COMMITS=$(git log --oneline --reverse)
    else
        log_info "Getting commits since $LAST_TAG"
        COMMITS=$(git log "${LAST_TAG}"..HEAD --oneline)
    fi
    
    if [[ -z "$COMMITS" ]]; then
        log_warning "No new commits found since last tag"
        return 0
    fi
    
    # Create changelog entry
    CHANGELOG_ENTRY="## [$NEW_VERSION] - $(date +%Y-%m-%d)

### Changes
$COMMITS

"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would add changelog entry:"
        echo -e "\n$CHANGELOG_ENTRY"
        return 0
    fi
    
    # Update CHANGELOG.md
    if [[ -f "CHANGELOG.md" ]]; then
        # Create temporary file with new entry + existing content
        {
            echo "# Changelog"
            echo ""
            echo "All notable changes to this project will be documented in this file."
            echo ""
            echo "$CHANGELOG_ENTRY"
            # Skip the header from existing changelog
            tail -n +5 CHANGELOG.md 2>/dev/null || echo ""
        } > CHANGELOG.md.tmp
        
        mv CHANGELOG.md.tmp CHANGELOG.md
    else
        # Create new CHANGELOG.md
        cat > CHANGELOG.md << EOF
# Changelog

All notable changes to this project will be documented in this file.

$CHANGELOG_ENTRY
EOF
    fi
    
    log_success "Changelog updated"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker build"
        return 0
    fi
    
    log_info "Building Docker images..."
    
    BUILD_ARGS=("--target" "production" "--version" "$NEW_VERSION")
    
    if [[ -n "$REGISTRY" ]]; then
        BUILD_ARGS+=("--registry" "$REGISTRY" "--push")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: ./scripts/build.sh ${BUILD_ARGS[*]}"
        return 0
    fi
    
    if [[ -x "./scripts/build.sh" ]]; then
        ./scripts/build.sh "${BUILD_ARGS[@]}"
    else
        log_warning "Build script not found or not executable. Skipping image build."
    fi
    
    log_success "Docker images built"
}

# Commit and tag
commit_and_tag() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would commit changes and create tag v$NEW_VERSION"
        return 0
    fi
    
    log_info "Committing changes and creating tag..."
    
    # Add changed files
    git add VERSION CHANGELOG.md pyproject.toml 2>/dev/null || true
    
    # Create commit
    git commit -m "chore: release v$NEW_VERSION

- Bump version to $NEW_VERSION
- Update changelog

🚀 Generated with release automation script" || {
        log_warning "No changes to commit"
    }
    
    # Create tag
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION

🚀 Automated release with:
- Version bump: $CURRENT_VERSION → $NEW_VERSION
- Release type: $RELEASE_TYPE
- Build date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')
- Commit: $(git rev-parse --short HEAD)"
    
    log_success "Tag v$NEW_VERSION created"
}

# Push to remote
push_to_remote() {
    if [[ "$PUSH_TAGS" != "true" ]]; then
        log_info "Skipping push to remote"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push commits and tags to remote"
        return 0
    fi
    
    log_info "Pushing to remote repository..."
    
    # Push commits
    git push origin HEAD
    
    # Push tags
    git push origin "v$NEW_VERSION"
    
    log_success "Changes pushed to remote"
}

# Create GitHub release
create_github_release() {
    if [[ "$CREATE_GITHUB_RELEASE" != "true" ]]; then
        log_info "Skipping GitHub release creation"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create GitHub release v$NEW_VERSION"
        return 0
    fi
    
    log_info "Creating GitHub release..."
    
    # Extract changelog entry for this version
    RELEASE_NOTES=""
    if [[ -f "CHANGELOG.md" ]]; then
        # Extract the section for this version
        RELEASE_NOTES=$(awk "/## \[$NEW_VERSION\]/,/## \[/{if(/## \[/ && !/## \[$NEW_VERSION\]/) exit; print}" CHANGELOG.md | head -n -1)
    fi
    
    if [[ -z "$RELEASE_NOTES" ]]; then
        RELEASE_NOTES="Release v$NEW_VERSION

Automated release created on $(date +'%Y-%m-%d %H:%M:%S UTC').

See CHANGELOG.md for detailed changes."
    fi
    
    # Create release
    gh release create "v$NEW_VERSION" \
        --title "Release v$NEW_VERSION" \
        --notes "$RELEASE_NOTES" \
        --latest
    
    log_success "GitHub release created"
}

# Generate release summary
generate_summary() {
    log_success "Release process completed!"
    
    echo
    echo "=================================="
    echo "  RELEASE SUMMARY"
    echo "=================================="
    echo "  Version: $CURRENT_VERSION → $NEW_VERSION"
    echo "  Type: $RELEASE_TYPE"
    echo "  Tag: v$NEW_VERSION"
    echo "  Dry run: $DRY_RUN"
    echo "=================================="
    echo
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Next steps:"
        echo "  • Verify the release: git show v$NEW_VERSION"
        echo "  • View on GitHub: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/releases/tag/v$NEW_VERSION"
        if [[ -n "$REGISTRY" ]]; then
            echo "  • Pull Docker image: docker pull $REGISTRY/rlhf-audit-trail:$NEW_VERSION"
        fi
        echo "  • Check CI/CD pipeline status"
    else
        log_info "This was a dry run. No changes were made."
        log_info "Run without --dry-run to perform the actual release."
    fi
}

# Main execution
main() {
    check_prerequisites
    get_current_version
    calculate_new_version
    run_tests
    update_version
    generate_changelog
    build_images
    commit_and_tag
    push_to_remote
    create_github_release
    generate_summary
}

# Execute main function
main "$@"