import os
import sys
import subprocess

class PackageInfo:
    """Parses general package info from setup.py and the git info"""

    def __init__(self, name, version, git_commit = None, python_version = None):
        self.name = name
        self.version = version
        self.python_version = python_version
        self._git_commit = git_commit

    @property
    def git_commit(self):
        if self._git_commit is None:
            label = subprocess.check_output(['git', 'describe', '--always']).strip()
            self._git_commit = label.decode('utf-8')

        return self._git_commit

    def __str__(self):
        
        args = []

        for arg_key, arg_value in self.__dict__.items():
            if arg_value is None: continue
            if arg_key.startswith('_'): continue
            args.append("%s=%s" % (arg_key, arg_value))

        return "PackageInfo(%s)" % ", ".join(args)


def parse_package_info():
    base_path = os.path.dirname(__file__)
    
    with open(os.path.join(base_path, "setup.py"), "r") as i:
        setup_info = i.read()
    
    name, version, pyversion = None, None, None

    for line in setup_info.splitlines():
        line = line.strip()
        if line.startswith("version"):
            _, version = line.split("=")
            version = version[1:-2]
        if line.startswith("name"):
            _, name  = line.split("=")
            name = name[1:-2]
        if line.startswith("python_requires"):
            _, pyversion  = line.split("=", 1)
            pyversion = pyversion[1:-2]
        
    return PackageInfo(name, version, python_version = pyversion)



def get_info():
    return parse_package_info()


def handle_script_info():
    # Print package information after start
    package_info = get_info()
    print("--- CST Transform %s [Git: %s] --------------------------------" % (package_info.version, package_info.git_commit))

    args = sys.argv
    if any(arg == "--version" for arg in args):
        
        print()
        print("CST Transform %s (Python %s)" % (package_info.version, package_info.python_version))

        exit()


if __name__ == "__main__":
    print(get_info())