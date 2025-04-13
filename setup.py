import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__="0.0.0"

REPO_NAME = "AutoSummaryAI"
AUTHOR_USER_NAME = "Venom567SR"
SRC_REPO = "AutoSummaryAI"
AUTHOR_EMAIL = "sahilrahate567@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for summarizing text using transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https:/github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https:/github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)

