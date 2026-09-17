import os, sys, subprocess

def run_tests():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = root
    
    # 1. Test CLI help
    r1 = subprocess.run([sys.executable, "-m", "bellium.cli", "--help"], cwd=root, env=env, capture_output=True, text=True)
    assert r1.returncode == 0
    print("CLI Help test: OK")
    
    # 2. Test CLI route
    r2 = subprocess.run([sys.executable, "-m", "bellium.cli", "route", "--tags", "cutout", "alpha"], cwd=root, env=env, capture_output=True, text=True)
    assert r2.returncode == 0
    assert "bellium/cutout" in r2.stdout
    print("CLI Route test: OK")
    
    # 3. Test CLI route with zero tokens
    r3 = subprocess.run([sys.executable, "-m", "bellium.cli", "route", "--tags", "inpaint", "--zero-tokens"], cwd=root, env=env, capture_output=True, text=True)
    assert r3.returncode == 0
    assert "bellium/inpaint_knn" in r3.stdout
    print("CLI Route zero tokens test: OK")
    
    print("All CLI tests PASSED!")

if __name__ == "__main__":
    run_tests()
