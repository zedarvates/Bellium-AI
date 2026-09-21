import os
import sys
import subprocess

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

    # 4. Test CLI filter with a generated image, then refuse source overwrite
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source.png")
        dst = os.path.join(tmp, "gray.png")
        Image.new("RGB", (8, 8), (200, 40, 40)).save(src)
        r4 = subprocess.run([sys.executable, "-m", "bellium.cli", "filter", "grayscale", src, "-o", dst], cwd=root, env=env, capture_output=True, text=True)
        assert r4.returncode == 0, r4.stderr
        assert os.path.exists(dst)
        with Image.open(dst) as out:
            pixel = out.convert("RGB").getpixel((0, 0))
            assert pixel[0] == pixel[1] == pixel[2]
        r5 = subprocess.run([sys.executable, "-m", "bellium.cli", "filter", "sepia", src, "-o", src], cwd=root, env=env, capture_output=True, text=True)
        assert r5.returncode == 2
        r6 = subprocess.run([sys.executable, "-m", "bellium.cli", "filter", "invert", src, "-o", os.path.join(tmp, "inv.png")], cwd=root, env=env, capture_output=True, text=True)
        assert r6.returncode == 0, r6.stderr
        with Image.open(os.path.join(tmp, "inv.png")) as inv:
            assert inv.convert("RGB").getpixel((0, 0)) == (55, 215, 215)
        r7 = subprocess.run([sys.executable, "-m", "bellium.cli", "filter", "sepia", src, "-o", os.path.join(tmp, "bad.png"), "--factor", "2"], cwd=root, env=env, capture_output=True, text=True)
        assert r7.returncode == 2
        print("CLI Filter test: OK")

        # 5. Test CLI texture repetition with a synthetic checkerboard and noise
        import random
        board = Image.new("L", (64, 64), 0)
        for y in range(64):
            for x in range(64):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    board.putpixel((x, y), 255)
        board_path = os.path.join(tmp, "board.png")
        board.save(board_path)
        r8 = subprocess.run([sys.executable, "-m", "bellium.cli", "texture", board_path], cwd=root, env=env, capture_output=True, text=True)
        assert r8.returncode == 0, r8.stderr
        assert "period=16 px" in r8.stdout, r8.stdout
        rng = random.Random(7)
        noise = Image.new("L", (64, 64))
        noise.putdata([rng.randrange(256) for _ in range(64 * 64)])
        noise_path = os.path.join(tmp, "noise.png")
        noise.save(noise_path)
        r9 = subprocess.run([sys.executable, "-m", "bellium.cli", "texture", noise_path], cwd=root, env=env, capture_output=True, text=True)
        assert r9.returncode == 2, r9.stdout
        print("CLI Texture test: OK")
    
    print("All CLI tests PASSED!")

if __name__ == "__main__":
    run_tests()
