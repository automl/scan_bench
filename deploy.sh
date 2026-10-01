uv version --bump patch      # (also: minor, major) we use semantic versioning: https://semver.org/
rm -rf dist/                 # clear old builds
uv build                     # creates the sdist and wheel in dist/
uv publish                   # uploads to PyPI