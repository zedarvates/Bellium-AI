"""Bundle the canonical model tree without moving the editable source models."""
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithModels(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / "models"
        for path in sorted(source.rglob("*.json")):
            target = Path(self.build_lib) / "bellium" / "_data" / "models" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(path), str(target))


setup(cmdclass={"build_py": BuildWithModels})
